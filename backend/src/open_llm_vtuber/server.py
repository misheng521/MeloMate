"""
MeloMate Server
========================
This module contains the WebSocket server for MeloMate, which handles
the WebSocket connections, serves static files, and manages the web tool.
It uses FastAPI for the server and Starlette for static file serving.
"""

import hmac
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles as StarletteStaticFiles

from .routes import init_client_ws_route, init_webtool_routes, init_proxy_route
from .service_context import ServiceContext
from .config_manager.utils import Config

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def frontend_origins() -> list[str]:
    configured = os.environ.get(
        "MELOMATE_FRONTEND_ORIGIN", "http://127.0.0.1:5178"
    )
    return [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]


# Create a custom StaticFiles class that adds CORS headers
class CORSStaticFiles(StarletteStaticFiles):
    """
    Static files handler that adds CORS headers to all responses.
    Needed because Starlette StaticFiles might bypass standard middleware.
    """

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)

        origin = dict(scope.get("headers") or []).get(b"origin", b"").decode(
            "latin-1"
        )
        if origin in frontend_origins():
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "X-MeloMate-Session"
        response.headers["Cross-Origin-Resource-Policy"] = "same-site"
        response.headers["X-Content-Type-Options"] = "nosniff"

        if path.endswith(".js"):
            response.headers["Content-Type"] = "application/javascript"

        return response


class WebSocketServer:
    """
    API server for MeloMate. This contains the websocket endpoint for the client, hosts the web tool, and serves static files.

    Creates and configures a FastAPI app, registers all routes
    (WebSocket, web tools, proxy) and mounts static assets with CORS.

    Args:
        config (Config): Application configuration containing system settings.
        default_context_cache (ServiceContext, optional):
            Pre‑initialized service context for sessions' service context to reference to.
            **If omitted, `initialize()` method needs to be called to load service context.**

    Notes:
        - If default_context_cache is omitted, call `await initialize()` to load service context cache.
        - Use `clean_cache()` to clear and recreate the local cache directory.
    """

    def __init__(self, config: Config, default_context_cache: ServiceContext = None):
        self.app = FastAPI(title="MeloMate Server")
        self.config = config
        self.default_context_cache = (
            default_context_cache or ServiceContext()
        )  # Use provided context or initialize a new empty one waiting to be loaded
        # It will be populated during the initialize method call

        # Add global CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=frontend_origins(),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "X-MeloMate-Session"],
        )

        @self.app.get("/api/health")
        async def health_check(request: Request):
            """Identify the backend instance to the local process supervisor."""
            expected = os.environ.get("MELOMATE_LAUNCH_TOKEN", "")
            supplied = request.headers.get("X-MeloMate-Launch", "")
            if not expected or not hmac.compare_digest(supplied, expected):
                return JSONResponse({"ok": False}, status_code=403)
            return JSONResponse(
                {
                    "ok": True,
                    "app": "MeloMate",
                    "service": "backend",
                },
                headers={
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                },
            )

        # Include routes, passing the context instance
        # The context will be populated during the initialize step
        self.app.include_router(
            init_client_ws_route(default_context_cache=self.default_context_cache),
        )
        self.app.include_router(
            init_webtool_routes(default_context_cache=self.default_context_cache),
        )

        # Initialize and include proxy routes if proxy is enabled
        system_config = config.system_config
        if hasattr(system_config, "enable_proxy") and system_config.enable_proxy:
            # Construct the server URL for the proxy
            host = system_config.host
            port = system_config.port
            server_url = f"ws://{host}:{port}/client-ws"
            self.app.include_router(
                init_proxy_route(server_url=server_url),
            )

        # Mount cache directory first (to ensure audio file access)
        if not os.path.exists("cache"):
            os.makedirs("cache")
        self.app.mount(
            "/cache",
            CORSStaticFiles(directory="cache"),
            name="cache",
        )

        # Mount static files with CORS-enabled handlers
        self.app.mount(
            "/live2d-models",
            CORSStaticFiles(directory=str(PROJECT_ROOT / "models" / "live2d")),
            name="live2d-models",
        )
        self.app.mount(
            "/bg",
            CORSStaticFiles(directory=str(PROJECT_ROOT / "backgrounds")),
            name="backgrounds",
        )
        if os.path.exists("web_tool"):
            self.app.mount(
                "/web-tool",
                CORSStaticFiles(directory="web_tool", html=True),
                name="web_tool",
            )

        if os.path.exists("frontend"):
            self.app.mount(
                "/",
                CORSStaticFiles(directory="frontend", html=True),
                name="frontend",
            )

    async def initialize(self):
        """Asynchronously load the service context from config.
        Calling this function is needed if default_context_cache was not provided to the constructor."""
        await self.default_context_cache.load_from_config(self.config)

    @staticmethod
    def clean_cache():
        """Clean the cache directory by removing and recreating it."""
        cache_dir = "cache"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)
