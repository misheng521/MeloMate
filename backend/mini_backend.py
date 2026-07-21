import atexit
import asyncio
import os
import sys
from pathlib import Path

import uvicorn
from loguru import logger

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
backend_models = ROOT.parent / "models" / "backend"
os.environ["HF_HOME"] = str(backend_models)
os.environ["MODELSCOPE_CACHE"] = str(backend_models)

from src.open_llm_vtuber.config_manager import load_config_with_character, validate_config
from src.open_llm_vtuber.server import WebSocketServer


def init_logger() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        "logs/mini_backend_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="14 days",
        level="DEBUG",
        encoding="utf-8",
    )


def main() -> None:
    init_logger()
    config = validate_config(load_config_with_character("conf.yaml"))
    server_config = config.system_config
    server = WebSocketServer(config=config)

    atexit.register(WebSocketServer.clean_cache)
    logger.info("Initializing MeloMate backend...")
    asyncio.run(server.initialize())
    logger.info(f"Backend listening on {server_config.host}:{server_config.port}")

    uvicorn.run(
        app=server.app,
        host=server_config.host,
        port=server_config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
