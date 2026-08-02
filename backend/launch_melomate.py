from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
FRONTEND_SERVER = PROJECT_ROOT / "server.mjs"
DEFAULT_FRONTEND_PORT = 5178
DEFAULT_WORKSPACE_PORT = 5179
DEFAULT_STARTUP_TIMEOUT_SECONDS = 600
HEALTH_TIMEOUT_SECONDS = 30


class StartupError(RuntimeError):
    pass


def parse_port(name: str, value: Any) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise StartupError(f"{name} must be an integer from 1 to 65535.") from exc
    if not 1 <= port <= 65535:
        raise StartupError(f"{name} must be an integer from 1 to 65535.")
    return port


def parse_timeout(value: Any) -> int:
    try:
        timeout = int(value)
    except (TypeError, ValueError) as exc:
        raise StartupError("MELOMATE_STARTUP_TIMEOUT must be an integer.") from exc
    if not 30 <= timeout <= 3600:
        raise StartupError("MELOMATE_STARTUP_TIMEOUT must be between 30 and 3600 seconds.")
    return timeout


def load_backend_endpoint() -> tuple[str, int, str]:
    config_path = BACKEND_ROOT / "conf.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        system_config = config["system_config"]
        host = str(system_config["host"]).strip()
        port = parse_port("system_config.port", system_config["port"])
    except StartupError:
        raise
    except Exception as exc:
        raise StartupError(f"Unable to read backend endpoint from {config_path}: {exc}") from exc

    if not host or any(character.isspace() for character in host) or any(
        character in host for character in "/\\?#"
    ):
        raise StartupError("system_config.host is invalid.")

    normalized_host = host.strip("[]")
    if normalized_host.lower() in {"localhost", "0.0.0.0", "::"}:
        connect_host = "127.0.0.1"
    else:
        connect_host = normalized_host
    uri_host = f"[{connect_host}]" if ":" in connect_host else connect_host
    return host, port, f"ws://{uri_host}:{port}/client-ws"


def frontend_port_from_environment() -> int:
    raw_port = (
        os.environ.get("MELOMATE_FRONTEND_PORT")
        or os.environ.get("PORT")
        or DEFAULT_FRONTEND_PORT
    )
    return parse_port("MELOMATE_FRONTEND_PORT", raw_port)


def workspace_port_from_environment(frontend_port: int) -> int:
    default_port = frontend_port + 1 if frontend_port < 65535 else DEFAULT_WORKSPACE_PORT
    raw_port = os.environ.get("MELOMATE_WORKSPACE_PORT") or default_port
    return parse_port("MELOMATE_WORKSPACE_PORT", raw_port)


def bind_probe_host(host: str) -> tuple[int, str]:
    normalized = host.strip("[]").lower()
    if normalized == "localhost":
        return socket.AF_INET, "127.0.0.1"
    if normalized == "::":
        return socket.AF_INET6, "::"
    if ":" in normalized:
        return socket.AF_INET6, host.strip("[]")
    return socket.AF_INET, host


def ensure_port_available(label: str, host: str, port: int, resolution: str) -> None:
    family, probe_host = bind_probe_host(host)
    probe = socket.socket(family, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind((probe_host, port))
    except OSError as exc:
        raise StartupError(
            f"{label} port {host}:{port} is already in use. "
            "MeloMate did not stop or reuse the owning process. "
            f"{resolution}"
        ) from exc
    finally:
        probe.close()


def process_exit_message(name: str, process: subprocess.Popen[Any]) -> str:
    return f"{name} exited during startup with code {process.returncode}."


def wait_for_frontend(
    url: str,
    launch_token: str,
    frontend: subprocess.Popen[Any],
    backend: subprocess.Popen[Any],
) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if frontend.poll() is not None:
            raise StartupError(process_exit_message("Frontend server", frontend))
        if backend.poll() is not None:
            raise StartupError(process_exit_message("Backend server", backend))
        try:
            request = urllib.request.Request(
                url, headers={"X-MeloMate-Launch": launch_token}
            )
            with opener.open(request, timeout=0.75) as response:
                payload = json.load(response)
            if (
                payload.get("app") == "MeloMate"
                and payload.get("service") == "frontend"
            ):
                return
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(0.2)
    raise StartupError("The frontend server did not pass its identity check within 30 seconds.")


def wait_for_backend(
    url: str,
    launch_token: str,
    timeout_seconds: int,
    frontend: subprocess.Popen[Any],
    backend: subprocess.Popen[Any],
) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if backend.poll() is not None:
            raise StartupError(process_exit_message("Backend server", backend))
        if frontend.poll() is not None:
            raise StartupError(process_exit_message("Frontend server", frontend))
        try:
            request = urllib.request.Request(
                url, headers={"X-MeloMate-Launch": launch_token}
            )
            with opener.open(request, timeout=0.75) as response:
                payload = json.load(response)
            if (
                payload.get("app") == "MeloMate"
                and payload.get("service") == "backend"
            ):
                return
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    raise StartupError(
        f"The backend did not pass its identity check within {timeout_seconds} seconds."
    )


def stop_owned_process(name: str, process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    print(f"Stopping MeloMate {name} process {process.pid}...")
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def monitor_processes(
    frontend: subprocess.Popen[Any], backend: subprocess.Popen[Any]
) -> int:
    while True:
        frontend_code = frontend.poll()
        backend_code = backend.poll()
        if frontend_code is not None:
            if frontend_code:
                print(f"[ERROR] Frontend server exited with code {frontend_code}.")
            return frontend_code or 0
        if backend_code is not None:
            print(f"[ERROR] Backend server exited with code {backend_code}.")
            return backend_code or 1
        time.sleep(0.5)


def main() -> int:
    frontend: subprocess.Popen[Any] | None = None
    backend: subprocess.Popen[Any] | None = None
    try:
        node = shutil.which("node")
        if not node:
            raise StartupError("Node.js was not found in PATH.")

        frontend_port = frontend_port_from_environment()
        workspace_port = workspace_port_from_environment(frontend_port)
        backend_bind_host, backend_port, backend_ws_url = load_backend_endpoint()
        if len({frontend_port, workspace_port, backend_port}) != 3:
            raise StartupError(
                "The frontend, isolated workspace, and backend must use different ports. "
                "Change MELOMATE_FRONTEND_PORT, MELOMATE_WORKSPACE_PORT, or system_config.port."
            )

        ensure_port_available(
            "Frontend",
            "127.0.0.1",
            frontend_port,
            "Close that application or set MELOMATE_FRONTEND_PORT to a free port.",
        )
        ensure_port_available(
            "Workspace",
            "127.0.0.1",
            workspace_port,
            "Close that application or set MELOMATE_WORKSPACE_PORT to a free port.",
        )
        ensure_port_available(
            "Backend",
            backend_bind_host,
            backend_port,
            "Close that application or change system_config.port in backend/conf.yaml.",
        )

        launch_token = secrets.token_urlsafe(32)
        session_token = secrets.token_urlsafe(32)
        frontend_url = f"http://127.0.0.1:{frontend_port}/"
        workspace_url = f"http://127.0.0.1:{workspace_port}"
        child_environment = os.environ.copy()
        child_environment["PORT"] = str(frontend_port)
        child_environment["MELOMATE_WORKSPACE_PORT"] = str(workspace_port)
        child_environment["MELOMATE_BACKEND_WS_URL"] = backend_ws_url
        child_environment["MELOMATE_LAUNCH_TOKEN"] = launch_token
        child_environment["MELOMATE_SESSION_TOKEN"] = session_token
        child_environment["MELOMATE_FRONTEND_ORIGIN"] = (
            f"{frontend_url.rstrip('/')},http://localhost:{frontend_port}"
        )
        child_environment["MELOMATE_FRONTEND_URL"] = frontend_url.rstrip("/")
        child_environment["MELOMATE_WORKSPACE_URL"] = workspace_url

        print(f"Starting MeloMate backend on {backend_bind_host}:{backend_port}...")
        backend = subprocess.Popen(
            [sys.executable, str(BACKEND_ROOT / "mini_backend.py")],
            cwd=str(PROJECT_ROOT),
            env=child_environment,
        )
        print(
            f"Starting MeloMate frontend on 127.0.0.1:{frontend_port} "
            f"with isolated workspace on 127.0.0.1:{workspace_port}..."
        )
        frontend = subprocess.Popen(
            [node, str(FRONTEND_SERVER)],
            cwd=str(PROJECT_ROOT),
            env=child_environment,
        )

        health_url = f"{frontend_url}api/health"
        wait_for_frontend(health_url, launch_token, frontend, backend)

        startup_timeout = parse_timeout(
            os.environ.get("MELOMATE_STARTUP_TIMEOUT", DEFAULT_STARTUP_TIMEOUT_SECONDS)
        )
        parsed_backend_url = urllib.parse.urlsplit(backend_ws_url)
        backend_health_host = parsed_backend_url.hostname or "127.0.0.1"
        backend_health_uri_host = (
            f"[{backend_health_host}]" if ":" in backend_health_host else backend_health_host
        )
        backend_health_url = f"http://{backend_health_uri_host}:{backend_port}/api/health"
        print("Waiting for the backend to finish initialization...")
        wait_for_backend(
            backend_health_url,
            launch_token,
            startup_timeout,
            frontend,
            backend,
        )

        print(f"MeloMate is ready at {frontend_url}")
        if os.environ.get("MELOMATE_NO_BROWSER") != "1":
            if not webbrowser.open(frontend_url):
                print(f"Open this address in your browser: {frontend_url}")
        return monitor_processes(frontend, backend)
    except KeyboardInterrupt:
        print("\nMeloMate was stopped by the user.")
        return 0
    except StartupError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    finally:
        stop_owned_process("frontend", frontend)
        stop_owned_process("backend", backend)


if __name__ == "__main__":
    raise SystemExit(main())
