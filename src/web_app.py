from __future__ import annotations

from dataclasses import dataclass

import tyro
from loguru import logger

from web_engine import EngineBusyError, EngineDecision, LunaEngineService, StrengthProfile
from web_factory import create_app
from web_games import ApiError, GameRecord, GameRegistry
from web_security import WebAppConfig, parse_exact_trusted_hosts

__all__ = [
    "ApiError",
    "EngineBusyError",
    "EngineDecision",
    "GameRecord",
    "GameRegistry",
    "LunaEngineService",
    "StrengthProfile",
    "WebAppConfig",
    "create_app",
    "parse_exact_trusted_hosts",
]


@dataclass
class WebServeConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    device: str = "cuda"
    checkpoint: str = "./runs/luna-main/latest.pth.tar"
    search_simulations: int = 96
    compile_inference: bool = True


def main() -> None:
    cfg = tyro.cli(WebServeConfig)
    try:
        engine = LunaEngineService(
            cfg.checkpoint,
            device=cfg.device,
            search_simulations=cfg.search_simulations,
            compile_inference=cfg.compile_inference,
        )
    except (FileNotFoundError, RuntimeError, ValueError, KeyError):
        logger.exception("Luna web server refused to start: the checkpoint could not be loaded")
        raise SystemExit(2) from None

    application = create_app(engine)
    logger.info("Luna web interface ready at http://{}:{}", cfg.host, cfg.port)
    application.run(host=cfg.host, port=cfg.port, debug=cfg.debug)


if __name__ == "__main__":
    main()
