"""
Flask application factory for the Goobusters distributed tracking server.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import Flask

# Add project root to Python path BEFORE any imports
# This allows running server/server.py directly
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lib.config import ServerConfig, load_config
from lib.mask_archive import mask_series_dir

# Now we can import server modules using absolute imports
# (they work because project_root is in sys.path)
from server.api.routes import create_api_blueprint
from server.storage.series_manager import SeriesManager
from server.startup import initialize_server

logger = logging.getLogger(__name__)


class ServerContext:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.series_manager = SeriesManager(config)


def cleanup_retrack_queue(
    config: ServerConfig,
    series_manager: SeriesManager,
    clear_queue: bool = False,
) -> None:
    """Clean up retrack queue on server startup."""
    from server.storage.retrack_queue import RetrackQueue

    queue_file = config.server_state_path / "retrack_queue.json"
    if not queue_file.exists():
        return

    retrack_queue = RetrackQueue(queue_file)

    if clear_queue:
        # Before clearing, revert any version IDs that were written by incomplete retrack jobs
        # If a retrack job was processing, it may have written new_version_id to frametype.json
        # We need to revert it back to the version that existed before the retrack started
        jobs = retrack_queue._load_queue()
        for job in jobs:
            # Only revert if job was processing (may have written version_id)
            # For pending jobs, frametype.json hasn't been updated yet
            if job.status == "processing":
                try:
                    from lib.mask_archive import mask_series_dir

                    output_dir = mask_series_dir(
                        Path(config.mask_storage_path),
                        config.flow_method,
                        job.study_uid,
                        job.series_uid,
                    )
                    # Check retrack frametype.json (where new_version_id would have been written)
                    retrack_frametype_path = (
                        output_dir / "retrack" / "frametype.json"
                    )
                    if retrack_frametype_path.exists():
                        # Remove retrack directory entirely (cleanup)
                        retrack_dir = output_dir / "retrack"
                        if retrack_dir.exists():
                            shutil.rmtree(retrack_dir)
                    # Revert main frametype.json to previous_version_id (the version before retrack started)
                    # This only matters if retrack completed and overwrote the main frametype.json
                    frametype_path = output_dir / "frametype.json"
                    if frametype_path.exists() and job.previous_version_id is not None:
                        try:
                            with frametype_path.open("r+") as f:
                                import json

                                data = json.load(f)
                                # Only revert if current version matches the new_version_id from the failed job
                                # This means the retrack completed but we're clearing it
                                if data.get("_version_id") == job.new_version_id:
                                    data["_version_id"] = job.previous_version_id
                                    f.seek(0)
                                    f.truncate()
                                    json.dump(data, f, indent=2)
                        except (json.JSONDecodeError, OSError):
                            pass
                except FileNotFoundError:
                    pass  # Series doesn't exist, skip

        # Cleanup: Remove old server_state/series/ directory (no longer used)
        # Version ID is now stored in output/{flow_method}/{study_uid}_{series_uid}/version.json
        series_root_old = config.server_state_path / "series"
        if series_root_old.exists():
            shutil.rmtree(series_root_old)
            logger.info(
                "Removed old server_state/series/ directory (migrated to filesystem-based storage)"
            )

        retrack_queue.clear_queue()
        print("Cleared retrack queue")
    else:
        reset_count = retrack_queue.reset_stale_processing_jobs()
        if reset_count > 0:
            print(
                f"Reset {reset_count} stale processing job(s) back to pending"
            )


def _run_retrack_worker_process(config: ServerConfig) -> None:
    """Process entrypoint for retrack worker (isolates OpenCV crashes from main server)."""
    from server.retrack_worker import worker_loop
    from server.storage.series_manager import SeriesManager
    import logging

    # Configure logging to append to server log/latest and also stdout
    log_dir = Path(__file__).parent / "log"
    latest_log = log_dir / "latest"
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(latest_log, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    series_manager = SeriesManager(config)
    worker_loop(config, series_manager)


def start_retrack_worker(
    config: ServerConfig, series_manager: SeriesManager
) -> None:
    """
    Start retrack worker in a separate process using spawn to isolate OpenCV crashes.
    """
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    worker_process = ctx.Process(
        target=_run_retrack_worker_process,
        args=(config,),
        daemon=True,
        name="retrack-worker",
    )
    worker_process.start()
    logger.info("Retrack worker process started (spawn)")


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

    # Start retrack worker in background thread
    if not skip_startup:
        start_retrack_worker(config, context.series_manager)

    return app


def get_pid_file(config: ServerConfig) -> Path:
    """Get path to PID file."""
    config.server_state_path.mkdir(parents=True, exist_ok=True)
    return config.server_state_path / "server.pid"


def _cleanup_server_state(config: ServerConfig, was_killed: bool) -> None:
    """Clean up old server_state files after successful startup."""
    state_path = config.server_state_path

    # Remove old series_index.json (index is now in-memory)
    old_index = state_path / "series_index.json"
    if old_index.exists():
        old_index.unlink()
        logger.info("Removed old series_index.json (index is now in-memory)")

    # Clean up uploaded_masks directory (temporary uploads, should be empty after retrack completes)
    uploaded_masks_dir = state_path / "uploaded_masks"
    if uploaded_masks_dir.exists():
        try:
            shutil.rmtree(uploaded_masks_dir)
            logger.info("Cleaned up uploaded_masks directory")
        except OSError as e:
            logger.warning(f"Failed to clean uploaded_masks directory: {e}")

    # Keep: retrack_queue.json (persists jobs across restarts)
    # Keep: server.pid (process management)


def read_pid(pid_file: Path) -> Optional[int]:
    """Read PID from file if it exists and process is still running."""
    if not pid_file.exists():
        return None

    try:
        pid = int(pid_file.read_text().strip())
        # Check if process is still running
        os.kill(pid, 0)  # Signal 0 doesn't kill, just checks if process exists
        return pid
    except (ValueError, OSError, ProcessLookupError):
        # PID file exists but process is dead - remove stale file
        pid_file.unlink(missing_ok=True)
        return None


def write_pid(pid_file: Path) -> None:
    """Write current process PID to file."""
    pid_file.write_text(str(os.getpid()))


def update_latest_log_symlink(log_file: Path) -> None:
    """Create or update symlink log/latest pointing to the current log file."""
    log_dir = log_file.parent
    latest_link = log_dir / "latest"

    # Remove existing symlink if it exists
    if latest_link.exists() or latest_link.is_symlink():
        try:
            latest_link.unlink()
        except OSError:
            pass  # Ignore errors removing old symlink

    # Create new symlink
    try:
        latest_link.symlink_to(log_file.name)
    except OSError as e:
        # Log but don't fail if symlink creation fails
        # Use print since logger might not be set up yet
        print(
            f"Warning: Failed to create latest log symlink: {e}",
            file=sys.stderr,
        )


def kill_existing(pid_file: Path) -> bool:
    """Kill existing process if PID file exists and process is running. Returns True if killed."""
    pid = read_pid(pid_file)
    if pid is None:
        return False

    try:
        os.kill(pid, signal.SIGTERM)
        # Wait a bit for graceful shutdown
        import time

        for _ in range(10):  # Wait up to 1 second
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except (OSError, ProcessLookupError):
                # Process is dead
                pid_file.unlink(missing_ok=True)
                return True
        # Still running, force kill
        os.kill(pid, signal.SIGKILL)
        pid_file.unlink(missing_ok=True)
        return True
    except (OSError, ProcessLookupError):
        # Process already dead
        pid_file.unlink(missing_ok=True)
        return False


def start_detached_process(
    script_path: Path,
    log_file: Path,
    pid_file: Path,
    kill_existing_flag: bool,
) -> None:
    """
    Start a detached child process without double-forking.

    Using subprocess with start_new_session avoids the Torch/MPS crash that
    happens when the original process forks after importing heavy libraries.
    """
    child_args: list[str] = []
    if kill_existing_flag:
        child_args.append("-k")
    cmd = [sys.executable, str(script_path), *child_args]

    # Ensure log file exists so the child can append immediately
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.touch(exist_ok=True)

    env = dict(os.environ)
    env["GOOBUSTERS_LOG_FILE"] = str(log_file)
    env["GOOBUSTERS_DAEMON_CHILD"] = "1"

    with open(os.devnull, "rb") as devnull, open(os.devnull, "ab", buffering=0) as devnull_out:
        subprocess.Popen(
            cmd,
            stdin=devnull,
            stdout=devnull_out,
            stderr=devnull_out,
            start_new_session=True,
            env=env,
        )

    print(f"Log file: {log_file}")
    print(f"PID file: {pid_file}")
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Goobusters tracking server")
    parser.add_argument(
        "-d",
        "--daemon",
        action="store_true",
        help="Run as daemon (detached process)",
    )
    parser.add_argument(
        "-k",
        "--kill",
        action="store_true",
        help="Kill existing server process before starting",
    )
    args = parser.parse_args()

    config = load_config("server")
    pid_file = get_pid_file(config)

    # Set up logging to both file and console
    # Log directory: log/ subdirectory from server.py location
    server_dir = Path(__file__).parent
    log_dir = server_dir / "log"
    log_dir.mkdir(exist_ok=True)

    # Allow daemon parent to pass a fixed log file to child to avoid churn
    env_log_file = os.environ.get("GOOBUSTERS_LOG_FILE")
    is_daemon_child = os.environ.get("GOOBUSTERS_DAEMON_CHILD") == "1"
    if env_log_file:
        log_file = Path(env_log_file)
    else:
        # Log filename: YYMMDD-HHMMSS.log (sortable, 24h format)
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        log_file = log_dir / f"{timestamp}.log"

    # Create/update symlink to latest log
    update_latest_log_symlink(log_file)

    # Kill existing process if requested
    if args.kill:
        if kill_existing(pid_file):
            print("Killed existing server process")
        else:
            print("No existing server process found")
        # Clear retrack queue when killing (stale jobs shouldn't be retried)
        # Need series_manager to revert version IDs
        from server.storage.series_manager import SeriesManager

        series_manager = SeriesManager(config)
        cleanup_retrack_queue(config, series_manager, clear_queue=True)

    # Check if server is already running
    existing_pid = read_pid(pid_file)
    if existing_pid is not None:
        print(f"Server is already running (PID: {existing_pid})")
        print(f"Use -k to kill it, or check PID file: {pid_file}")
        sys.exit(1)

    # Detach early to avoid fork-with-torch crashes
    if args.daemon:
        start_detached_process(
            Path(__file__).resolve(), log_file, pid_file, args.kill
        )

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

    # Console handler - only if stdout is a real console (avoid double-writing to log file in daemon mode)
    if not is_daemon_child:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # Write PID file
    write_pid(pid_file)

    # Clean up stale retrack queue jobs on startup (if not already done with -k)
    if not args.kill:
        from server.storage.series_manager import SeriesManager

        series_manager = SeriesManager(config)
        cleanup_retrack_queue(config, series_manager, clear_queue=False)

    # Log startup message with log file location
    logger.info("=" * 60)
    logger.info("Starting Goobusters Server")
    logger.info(f"PID: {os.getpid()}")
    logger.info(f"PID file: {pid_file}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)

    try:
        app = create_app(config, skip_startup=False)

        # Clean up old server_state files after successful startup
        _cleanup_server_state(config, args.kill)

        logger.info(
            f"Server ready on {config.server_host}:{config.server_port}"
        )
        app.run(host=config.server_host, port=config.server_port)
    finally:
        # Clean up PID file on exit
        pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
