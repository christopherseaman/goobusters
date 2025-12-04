"""
Flask application factory for the Goobusters distributed tracking server.
"""

from __future__ import annotations

import logging
from typing import Optional

from flask import Flask

from lib.config import ServerConfig, load_config
from server.api.routes import create_api_blueprint
from server.storage.series_manager import SeriesManager


class ServerContext:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.series_manager = SeriesManager(config)


def create_app(config: Optional[ServerConfig] = None) -> Flask:
    config = config or load_config("server")
    context = ServerContext(config)

    app = Flask(__name__)
    app.config["SERVER_CONTEXT"] = context

    api_bp = create_api_blueprint(context.series_manager, config)
    app.register_blueprint(api_bp)

    @app.route("/healthz")
    def healthcheck():
        return {"ok": True}

    return app


def main() -> None:
    config = load_config("server")
    app = create_app(config)
    logging.basicConfig(level=logging.INFO)
    app.run(host=config.server_host, port=config.server_port)


if __name__ == "__main__":
    main()
