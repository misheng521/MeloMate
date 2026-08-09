import hmac
import os
from uuid import uuid4
from fastapi import APIRouter, WebSocket, Response
from starlette.websockets import WebSocketDisconnect
from loguru import logger
from .service_context import ServiceContext
from .websocket_handler import WebSocketHandler

SESSION_PROTOCOL_PREFIX = "melomate.session."


def _allowed_frontend_origins() -> set[str]:
    configured = os.environ.get(
        "MELOMATE_FRONTEND_ORIGIN", "http://127.0.0.1:5178"
    )
    return {
        origin.strip().rstrip("/")
        for origin in configured.split(",")
        if origin.strip()
    }


def _authenticated_websocket_protocol(websocket: WebSocket) -> str | None:
    expected = os.environ.get("MELOMATE_SESSION_TOKEN", "")
    if not expected:
        return None

    origin = websocket.headers.get("origin", "").rstrip("/")
    if origin and origin not in _allowed_frontend_origins():
        return None

    expected_protocol = f"{SESSION_PROTOCOL_PREFIX}{expected}"
    protocols = {
        value.strip()
        for value in websocket.headers.get("sec-websocket-protocol", "").split(",")
        if value.strip()
    }
    for protocol in protocols:
        if hmac.compare_digest(protocol, expected_protocol):
            return protocol
    return None


async def _accept_authenticated_websocket(websocket: WebSocket) -> bool:
    protocol = _authenticated_websocket_protocol(websocket)
    if not protocol:
        await websocket.close(code=1008)
        return False
    await websocket.accept(subprotocol=protocol)
    return True


def init_client_ws_route(default_context_cache: ServiceContext) -> APIRouter:
    """
    Create and return API routes for handling the `/client-ws` WebSocket connections.

    Args:
        default_context_cache: Default service context cache for new sessions.

    Returns:
        APIRouter: Configured router with WebSocket endpoint.
    """

    router = APIRouter()
    ws_handler = WebSocketHandler(default_context_cache)

    @router.websocket("/client-ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for client connections"""
        if not await _accept_authenticated_websocket(websocket):
            return
        client_uid = str(uuid4())

        try:
            await ws_handler.handle_new_connection(websocket, client_uid)
            await ws_handler.handle_websocket_communication(websocket, client_uid)
        except WebSocketDisconnect:
            await ws_handler.handle_disconnect(client_uid)
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}")
            await ws_handler.handle_disconnect(client_uid)
            raise

    return router


def init_proxy_route(server_url: str) -> APIRouter:
    """
    Create and return API routes for handling proxy connections.

    Args:
        server_url: The WebSocket URL of the actual server

    Returns:
        APIRouter: Configured router with proxy WebSocket endpoint
    """
    router = APIRouter()
    from .proxy_handler import ProxyHandler

    proxy_handler = ProxyHandler(server_url)

    @router.websocket("/proxy-ws")
    async def proxy_endpoint(websocket: WebSocket):
        """WebSocket endpoint for proxy connections"""
        if not _authenticated_websocket_protocol(websocket):
            await websocket.close(code=1008)
            return
        try:
            await proxy_handler.handle_client_connection(websocket)
        except Exception as e:
            logger.error(f"Error in proxy connection: {e}")
            raise

    return router


def init_webtool_routes(default_context_cache: ServiceContext) -> APIRouter:
    """
    Create and return API routes for handling web tool interactions.

    Args:
        default_context_cache: Default service context cache for new sessions.

    Returns:
        APIRouter: Configured router with WebSocket endpoint.
    """

    router = APIRouter()

    @router.get("/web-tool")
    async def web_tool_redirect():
        """Redirect /web-tool to /web_tool/index.html"""
        return Response(status_code=302, headers={"Location": "/web-tool/index.html"})

    @router.get("/web_tool")
    async def web_tool_redirect_alt():
        """Redirect /web_tool to /web_tool/index.html"""
        return Response(status_code=302, headers={"Location": "/web-tool/index.html"})

    return router
