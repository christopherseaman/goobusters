"""
Flask application factory for the Goobusters distributed tracking server.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

from flask import Flask

# Add project root to Python path BEFORE any imports
# This allows running server/server.py directly
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lib.config import ServerConfig, load_config

# Now we can import server modules using absolute imports
# (they work because project_root is in sys.path)
from server.api.routes import create_api_blueprint
from server.storage.series_manager import SeriesManager
from server.startup import initialize_server

logger = logging.getLogger(__name__)


class MDaiOutputCapture:
    """Capture MD.ai SDK stdout output and redirect to logger."""
    
    def __init__(self, original_stream, logger):
        self.original_stream = original_stream
        self.logger = logger
    
    def write(self, text):
        # Handle both bytes and strings (Flask's click.echo may write bytes)
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        
        if text.strip():  # Only log non-empty lines
            for line in text.rstrip().split('\n'):
                if line.strip():
                    self.logger.info(f"[MD.ai SDK] {line}")
        # Don't write to original stream - we've already logged it
        # This prevents triple logging
    def flush(self):
        self.original_stream.flush()


class ServerContext:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.series_manager = SeriesManager(config)


def create_app(
    config: Optional[ServerConfig] = None,
    skip_startup: bool = False,
) -> Flask:
    """
    Create Flask application.

    Args:
        config: Server configuration (loads from env if None)
        skip_startup: If True, skip startup initialization (for testing)
    """
    config = config or load_config("server")
    context = ServerContext(config)

    # Initialize server: download dataset and generate masks
    # This must complete before serving clients (per DISTRIBUTED_ARCHITECTURE.md)
    if not skip_startup:
        initialize_server(config, context.series_manager)

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
    
    # Set up logging to both file and console
    # Log directory: log/ subdirectory from server.py location
    server_dir = Path(__file__).parent
    log_dir = server_dir / "log"
    log_dir.mkdir(exist_ok=True)
    
    # Log filename: YYMMDD-HHMMSS.log (sortable, 24h format)
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    log_file = log_dir / f"{timestamp}.log"
    
    # Configure root logger with file and console handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Store original stdout before we replace it
    original_stdout = sys.stdout

    # Capture MD.ai SDK output and redirect to logger
    # MD.ai SDK prints directly to stdout, so we intercept it
    mdai_capture = MDaiOutputCapture(original_stdout, logger)
    sys.stdout = mdai_capture

    # Console handler - write to original stdout to avoid recursion
    console_handler = logging.StreamHandler(original_stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Log startup message with log file location
    logger.info("=" * 60)
    logger.info("Starting Goobusters Server")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)
    
    try:
        app = create_app(config, skip_startup=False)
        logger.info(f"Server ready on {config.server_host}:{config.server_port}")
        app.run(host=config.server_host, port=config.server_port)
    finally:
        # Restore original stdout
        sys.stdout = original_stdout



if __name__ == "__main__":
    main()
