"""
Flask application factory for the Goobusters distributed tracking server.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
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
import threading

logger = logging.getLogger(__name__)


class MDaiOutputCapture:
    """Capture MD.ai SDK stdout output and redirect to logger."""

    def __init__(self, original_stream, logger):
        self.original_stream = original_stream
        self.logger = logger

    def write(self, text):
        # Handle both bytes and strings (Flask's click.echo may write bytes)
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")

        if text.strip():  # Only log non-empty lines
            for line in text.rstrip().split("\n"):
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
        # Before clearing, revert any temporary version IDs back to previous versions
        jobs = retrack_queue._load_queue()
        for job in jobs:
            if job.previous_version_id is not None:
                try:
                    series_manager.update_version(
                        job.study_uid,
                        job.series_uid,
                        job.previous_version_id,
                        job.editor,
                    )
                except FileNotFoundError:
                    pass  # Series metadata doesn't exist, skip

        # After clearing queue, fix any orphaned version IDs
        # If a series has a version ID but no active job, check if it matches the archive
        # If not, sync it from the archive (archive is source of truth)
        import tarfile
        import json
        from pathlib import Path

        mask_root = Path(config.mask_storage_path)
        flow_method = config.flow_method

        # Check all series metadata for orphaned version IDs
        for metadata_file in series_manager.series_root.glob("*/metadata.json"):
            try:
                with metadata_file.open() as f:
                    metadata_data = json.load(f)
                    study_uid = metadata_data.get("study_uid")
                    series_uid = metadata_data.get("series_uid")
                    current_version_id = metadata_data.get("current_version_id")

                    if not study_uid or not series_uid:
                        continue

                    # Check if there's an active job for this series
                    has_active_job = retrack_queue.has_active_job(
                        study_uid, series_uid
                    )
                    if has_active_job:
                        continue  # Skip, job will handle version

                    # Check if version matches archive (archive is source of truth)
                    output_dir = (
                        mask_root / flow_method / f"{study_uid}_{series_uid}"
                    )
                    retrack_archive = output_dir / "retrack" / "masks.tar"
                    initial_archive = output_dir / "masks.tar"

                    archive_path = (
                        retrack_archive
                        if retrack_archive.exists()
                        else (
                            initial_archive
                            if initial_archive.exists()
                            else None
                        )
                    )

                    if archive_path:
                        # Read version from archive and sync if different (archive is source of truth)
                        try:
                            with tarfile.open(archive_path, "r") as tar:
                                metadata_file_obj = tar.extractfile(
                                    "metadata.json"
                                )
                                if metadata_file_obj:
                                    archive_metadata = json.load(
                                        metadata_file_obj
                                    )
                                    archive_version = archive_metadata.get(
                                        "version_id"
                                    )
                                    # Sync version from archive (even if archive_version is None/null)
                                    if archive_version != current_version_id:
                                        if archive_version:
                                            # Archive has a version - update metadata
                                            series_manager.update_version(
                                                study_uid,
                                                series_uid,
                                                archive_version,
                                                metadata_data.get("last_editor")
                                                or "system",
                                            )
                                        else:
                                            # Archive has null version - clear metadata version
                                            with metadata_file.open("r+") as f:
                                                metadata_data = json.load(f)
                                                metadata_data[
                                                    "current_version_id"
                                                ] = None
                                                f.seek(0)
                                                f.truncate()
                                                json.dump(
                                                    metadata_data, f, indent=2
                                                )
                        except Exception:
                            pass  # Archive might be corrupt, skip
                    elif current_version_id:
                        # No archive exists but version ID is set - clear it (orphaned from failed/cleared retrack)
                        # We need to directly modify the metadata file since update_version doesn't accept None
                        with metadata_file.open("r+") as f:
                            metadata_data = json.load(f)
                            metadata_data["current_version_id"] = None
                            f.seek(0)
                            f.truncate()
                            json.dump(metadata_data, f, indent=2)
            except Exception:
                pass  # Skip invalid metadata files

        retrack_queue.clear_queue()
        print("Cleared retrack queue and fixed orphaned version IDs")
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


def daemonize(log_file: Path) -> None:
    """Detach process and run in background."""
    try:
        pid = os.fork()
        if pid > 0:
            # Parent process - print log location and exit
            print(f"Log file: {log_file}")
            print(f"PID file: {get_pid_file(load_config('server'))}")
            os._exit(0)
    except OSError as e:
        sys.stderr.write(f"Fork failed: {e}\n")
        sys.exit(1)

    # Child process continues
    os.setsid()  # Create new session

    # Second fork
    try:
        pid = os.fork()
        if pid > 0:
            # Parent process - exit
            os._exit(0)
    except OSError as e:
        sys.stderr.write(f"Second fork failed: {e}\n")
        sys.exit(1)

    # Redirect standard file descriptors to /dev/null
    os.chdir("/")
    os.umask(0)

    # Redirect stdin, stdout, stderr to /dev/null
    with open("/dev/null", "r") as devnull:
        os.dup2(devnull.fileno(), sys.stdin.fileno())
    with open("/dev/null", "w") as devnull:
        os.dup2(devnull.fileno(), sys.stdout.fileno())
        os.dup2(devnull.fileno(), sys.stderr.fileno())

    # Close file descriptors (except 0, 1, 2 which are now /dev/null)
    import resource

    maxfd = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    if maxfd == resource.RLIM_INFINITY:
        maxfd = 1024

    for fd in range(3, maxfd):
        try:
            os.close(fd)
        except OSError:
            pass


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

    # Daemonize if requested (must be before logging setup)
    if args.daemon:
        daemonize(log_file)

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
        logger.info(
            f"Server ready on {config.server_host}:{config.server_port}"
        )
        app.run(host=config.server_host, port=config.server_port)
    finally:
        # Clean up PID file on exit
        pid_file.unlink(missing_ok=True)
        # Restore original stdout
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()
