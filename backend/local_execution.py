"""AutoGen-style local subprocess execution using installed runtimes.

Commands run as the current user. This is not a filesystem/network sandbox.
The model writes code with workspace tools, then runs that original file here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

import workspace_core as workspace

OUTPUT_LIMIT = 64000
HIDDEN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_venv_lock = threading.Lock()
# EOF refuses execution if the owner exits before releasing the gate. The
# project command starts only after the helper belongs to its process tree.
_GATE = "import json,subprocess,sys; token=sys.stdin.buffer.read(1); sys.exit(subprocess.call(json.loads(sys.argv[1]),stdin=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)) if token==b'1' else 125)"


def base_python() -> str:
    return str(getattr(sys, "_base_executable", None) or sys.executable)


def runtime_info() -> dict:
    python = base_python()
    return {"available": Path(python).is_file(), "runtime": "local", "platform": sys.platform,
            "python": python, "node": shutil.which("node"), "npm": shutil.which("npm"),
            "powershell": shutil.which("pwsh") or shutil.which("powershell"),
            "filesystem_isolated": False, "network_isolated": False,
            "permissions": "current user", "max_timeout_seconds": 600,
            "reason": "Runs workspace code files with installed Python/Node.js in hidden child processes. No extra execution service is required.",
            "dependencies": "Python uses installed Python or cwd/.venv when present. pip prepares a project venv offline if needed. npm uses project node_modules. Keep cwd unchanged to reuse dependencies.",
            "boundary": "File tools and starting cwd are workspace-scoped. Executed code has current-user file and network access, including outside the workspace."}


def project_environment(root: Path) -> dict[str, str]:
    # Keep backend API keys, launch tokens and runtime injection hooks out of the
    # child's inherited environment. This does not hide files from local code.
    env = {k: v for k, v in os.environ.items() if k.upper() in
           {"SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "PATH", "LANG", "LC_ALL",
            "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432", "PROGRAMDATA"}}
    home = root / ".runtime"
    if workspace._is_reparse_point(home):
        raise ValueError("Project runtime directory must not be a link")
    home.mkdir(exist_ok=True)
    scripts = root / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    env.update({"HOME": str(home), "USERPROFILE": str(home), "TEMP": str(home), "TMP": str(home),
                "APPDATA": str(home), "LOCALAPPDATA": str(home),
                "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                "PYTHONNOUSERSITE": "1", "PIP_CONFIG_FILE": os.devnull,
                "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_CACHE_DIR": str(home / "pip-cache"),
                "npm_config_cache": str(home / "npm-cache"),
                "npm_config_userconfig": str(home / "npmrc"),
                "npm_config_prefix": str(home / "npm-global"),
                "PATH": str(scripts) + os.pathsep + env.get("PATH", "")})
    return env


def _execute(argv: list[str], root: Path, env: dict, timeout: float, cancel: threading.Event) -> dict:
    if cancel.is_set():
        return {"ok": False, "executed": False, "cancelled": True}
    output = bytearray()
    total = 0
    process = reader = job = None
    timed_out = cancelled = cleanup_failed = executed = False
    error = None
    started = time.monotonic()

    def drain():
        nonlocal total
        try:
            while chunk := process.stdout.read1(4096):
                total += len(chunk)
                output.extend(chunk)
                del output[:-OUTPUT_LIMIT]
        except (OSError, ValueError):
            pass

    try:
        if os.name == "nt":
            from windows_process_job import WindowsJob
            job = WindowsJob()
        process = subprocess.Popen([base_python(), "-I", "-u", "-c", _GATE, json.dumps(argv)],
                                   cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, creationflags=HIDDEN,
                                   start_new_session=os.name != "nt")
        if job:
            job.assign(process)
        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        if not cancel.is_set():
            process.stdin.write(b"1")
            process.stdin.flush()
            executed = True
        process.stdin.close()
        while process.poll() is None:
            if cancel.is_set():
                cancelled = True
                break
            if time.monotonic() - started >= timeout:
                timed_out = True
                break
            try:
                process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                pass
        cancelled = cancelled or cancel.is_set()
    except (OSError, subprocess.SubprocessError) as exc:
        error = str(exc)
    finally:
        # Also reap descendants after a successful parent exit.
        try:
            if job:
                job.close()
            elif process and os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
        except (OSError, subprocess.SubprocessError):
            cleanup_failed = True
        if process:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if reader:
                reader.join(timeout=3)
                if reader.is_alive():
                    cleanup_failed = True
            if not reader or not reader.is_alive():
                process.stdout.close()
    result = {"ok": executed and not any((error, timed_out, cancelled, cleanup_failed)) and process.returncode == 0,
              "executed": executed, "runtime": "local", "cwd": str(root),
              "exit_code": process.returncode if process else None, "timed_out": timed_out,
              "cancelled": cancelled, "cleanup_failed": cleanup_failed,
              "output": output.decode("utf-8", errors="replace"), "output_truncated": total > OUTPUT_LIMIT,
              "output_retained": "last 64000 bytes", "filesystem_isolated": False, "network_isolated": False}
    if error:
        result["error"] = error
    return result


def _python(root: Path, env: dict, deadline: float, cancel: threading.Event, need_pip: bool) -> tuple[str | None, dict | None]:
    target = root / ".venv"
    scripts = target / ("Scripts" if os.name == "nt" else "bin")
    executable = scripts / ("python.exe" if os.name == "nt" else "python")
    if not target.exists() and not need_pip:
        return base_python(), None
    for path in (target, scripts):
        if workspace._is_reparse_point(path):
            raise ValueError("Project Python environment directory must not be a link")
    while not _venv_lock.acquire(timeout=0.05):
        if cancel.is_set() or time.monotonic() >= deadline:
            return None, {"ok": False, "executed": False, "cancelled": cancel.is_set(), "timed_out": not cancel.is_set()}
    try:
        if cancel.is_set() or time.monotonic() >= deadline:
            return None, {"ok": False, "executed": False, "cancelled": cancel.is_set(), "timed_out": not cancel.is_set()}
        if executable.is_file() and (target / "pyvenv.cfg").is_file():
            env["VIRTUAL_ENV"] = str(target)
            return str(executable), None
        if target.exists() and any(target.iterdir()):
            raise ValueError("Existing .venv is incomplete or belongs to another OS. Rename it before creating a new environment; existing files were not replaced.")
        prepared = _execute([base_python(), "-I", "-m", "venv", "--copies", str(target)],
                            root, env, deadline - time.monotonic(), cancel)
        if not prepared["ok"]:
            return None, {**prepared, "executed": False, "phase": "prepare_python",
                          "error": "Project Python preparation failed; requested code has not run. See output."}
        env["VIRTUAL_ENV"] = str(target)
        return str(executable), None
    finally:
        _venv_lock.release()


def _node_command(argv: list[str], root: Path) -> list[str]:
    node = shutil.which("node")
    if not node:
        raise ValueError("Node.js is not available. Use the Node.js installed during MeloMate deployment.")
    # Give this project its own package boundary: otherwise Node/npm can inherit
    # MeloMate's type=module and package configuration from an ancestor folder.
    try:
        with (root / "package.json").open("x", encoding="utf-8") as package:
            package.write('{"private": true}\n')
    except FileExistsError:
        pass  # The model's existing project configuration always takes priority.
    name = argv[0].lower()
    if name.startswith(("npm", "npx")):
        cli = "npm-cli.js" if name.startswith("npm") else "npx-cli.js"
        npm = shutil.which("npm") or shutil.which("npm.cmd")
        candidates = [Path(node).parent / "node_modules/npm/bin" / cli]
        if npm:
            candidates += [Path(npm).parent / "node_modules/npm/bin" / cli, Path(npm).resolve().parent / cli]
        script = next((p for p in candidates if p.is_file()), None)
        if not script:
            raise ValueError("The installed npm CLI could not be located. Repair the existing Node.js installation.")
        return [node, str(script), *argv[1:]]
    return [node, *argv[1:]]


def run_command(persona: str, argv: list[str], cwd: str = "", timeout_seconds: int = 120,
                cancel_event: threading.Event | None = None) -> dict:
    cancel = cancel_event or threading.Event()
    try:
        if not isinstance(argv, list) or not argv or len(argv) > 100 or any(not isinstance(v, str) or "\0" in v or len(v) > 16000 for v in argv) or not argv[0]:
            raise ValueError("argv must be a nonempty list of bounded strings")
        root = workspace.workspace_path(persona, cwd)
        if not root.is_dir():
            raise ValueError("Working directory does not exist; create it with workspace tools first.")
        deadline = time.monotonic() + min(600, max(1, int(timeout_seconds)))
        if cancel.is_set():
            return {"ok": False, "executed": False, "cancelled": True}
        env = project_environment(root)
        command = list(argv)
        name = command[0].lower()
        if name in {"python", "python3", "python.exe", "py", "pip", "pip3", "pip.exe"}:
            pip_alias = name.startswith("pip")
            need_pip = pip_alias or command[1:3] == ["-m", "pip"]
            python, failure = _python(root, env, deadline, cancel, need_pip)
            if failure:
                return failure
            command = [python, *(["-m", "pip"] if pip_alias else []), *command[1:]]
        elif name in {"node", "node.exe", "npm", "npm.cmd", "npx", "npx.cmd"}:
            command = _node_command(command, root)
        else:
            candidate = Path(command[0])
            if not candidate.is_absolute():
                candidate = root / candidate
            resolved = str(candidate) if candidate.is_file() else shutil.which(command[0], path=env["PATH"])
            if not resolved:
                raise ValueError("Command executable was not found: " + command[0])
            command[0] = str(Path(resolved).resolve())
            if os.name == "nt" and Path(command[0]).suffix.lower() in {".cmd", ".bat"}:
                raise ValueError("Run batch files through an explicit cmd.exe /d /c command. npm/npx are handled directly.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {"ok": False, "executed": False, "timed_out": True, "phase": "prepare"}
        return _execute(command, root, env, remaining, cancel)
    except (OSError, ValueError) as exc:
        return {"ok": False, "executed": False, "runtime": "local", "error": str(exc)}


async def run_command_async(*args, **kwargs) -> dict:
    """Propagate MCP cancellation to the worker and wait for process cleanup."""
    import anyio
    cancel, started, finished = threading.Event(), threading.Event(), threading.Event()
    def work():
        started.set()
        try:
            return run_command(*args, **kwargs, cancel_event=cancel)
        finally:
            finished.set()
    try:
        return await anyio.to_thread.run_sync(work, abandon_on_cancel=True)
    finally:
        cancel.set()
        with anyio.CancelScope(shield=True):
            with anyio.move_on_after(15):
                while started.is_set() and not finished.is_set():
                    await anyio.sleep(0.05)
