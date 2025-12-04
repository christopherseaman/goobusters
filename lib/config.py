"""
Centralised configuration loader shared by server and client processes.

Loads the root `dot.env` alongside optional role-specific overrides
(`dot.env.server`, `dot.env.client`). Values are coerced into strongly typed
dataclasses so downstream code never needs to parse raw environment strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from dotenv import dotenv_values


class ConfigError(RuntimeError):
    """Raised when a required configuration value is missing."""


@dataclass(frozen=True)
class SharedConfig:
    mdai_token: str
    data_dir: Path
    domain: str
    project_id: str
    dataset_id: str
    label_id: str
    empty_id: str
    flow_method: str
    test_study_uid: Optional[str] = None
    test_series_uid: Optional[str] = None


@dataclass(frozen=True)
class ServerConfig(SharedConfig):
    server_host: str = "0.0.0.0"
    server_port: int = 5000
    server_state_path: Path = Path("server_state")
    mask_storage_path: Path = Path("output")
    recent_view_threshold_minutes: int = 60
    retrack_workers: int = 2
    flow_method: str = "dis"


@dataclass(frozen=True)
class ClientConfig(SharedConfig):
    client_port: int = 8080
    server_url: str = "http://localhost:5000"
    user_email: Optional[str] = None
    video_cache_path: Path = Path("client_cache/data")


def _coerce_bool(value: Optional[str]) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


def _resolve_env(base_path: Path, role: Literal["shared", "server", "client"]) -> dict[str, str]:
    """
    Load environment dictionaries in priority order:
    1. `dot.env`
    2. `dot.env.{role}` (if present)
    3. Real environment (for overrides when running in production)
    """
    env: dict[str, str] = {}

    root_file = base_path / "dot.env"
    if root_file.exists():
        env.update({k: v for k, v in dotenv_values(root_file).items() if v is not None})

    role_file = base_path / f"dot.env.{role}"
    if role != "shared" and role_file.exists():
        env.update({k: v for k, v in dotenv_values(role_file).items() if v is not None})

    # Finally, overlay os.environ without importing os at module level to keep
    # import time fast for tools that only need metadata.
    import os

    env.update({k: v for k, v in os.environ.items() if v is not None})
    return env


def load_config(role: Literal["server", "client", "shared"] = "shared") -> SharedConfig:
    """
    Build the typed configuration for the requested role.
    """

    base_path = Path(".").resolve()
    raw = _resolve_env(base_path, role)

    required = [
        "MDAI_TOKEN",
        "DATA_DIR",
        "DOMAIN",
        "PROJECT_ID",
        "DATASET_ID",
        "LABEL_ID",
        "EMPTY_ID",
    ]
    missing = [key for key in required if not raw.get(key)]
    if missing:
        raise ConfigError(f"Missing required config keys: {', '.join(missing)}")

    shared_kwargs = dict(
        mdai_token=raw["MDAI_TOKEN"],
        data_dir=Path(raw["DATA_DIR"]).expanduser().resolve(),
        domain=raw["DOMAIN"],
        project_id=raw["PROJECT_ID"],
        dataset_id=raw["DATASET_ID"],
        label_id=raw["LABEL_ID"],
        empty_id=raw["EMPTY_ID"],
        flow_method=raw.get("FLOW_METHOD", "dis"),
        test_study_uid=raw.get("TEST_STUDY_UID"),
        test_series_uid=raw.get("TEST_SERIES_UID"),
    )

    if role == "server":
        server_kwargs = dict(
            server_host=raw.get("SERVER_HOST", "0.0.0.0"),
            server_port=int(raw.get("SERVER_PORT", 5000)),
            server_state_path=Path(raw.get("SERVER_STATE_PATH", "server_state")).resolve(),
            mask_storage_path=Path(raw.get("MASK_STORAGE_PATH", "output")).resolve(),
            recent_view_threshold_minutes=int(raw.get("RECENT_VIEW_THRESHOLD_MINUTES", 60)),
            retrack_workers=int(raw.get("RETRACK_WORKERS", 2)),
            flow_method=raw.get("FLOW_METHOD", "dis"),
        )
        return ServerConfig(**shared_kwargs, **server_kwargs)

    if role == "client":
        client_kwargs = dict(
            client_port=int(raw.get("CLIENT_PORT", 8080)),
            server_url=raw.get("SERVER_URL", "http://localhost:5000"),
            user_email=raw.get("USER_EMAIL"),
            video_cache_path=Path(raw.get("VIDEO_CACHE_PATH", "client_cache/data")).resolve(),
        )
        return ClientConfig(**shared_kwargs, **client_kwargs)

    return SharedConfig(**shared_kwargs)
