"""
Centralised configuration loader shared by server and client processes.

Loads `dot.yaml` for configuration. Values are coerced into strongly typed
dataclasses so downstream code never needs to parse raw config values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import yaml


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


@dataclass(frozen=True)
class ClientConfig(SharedConfig):
    client_port: int = 8080
    server_url: str = "http://localhost:5000"
    user_email: Optional[str] = None
    video_cache_path: Path = Path("client_cache/data")
    frames_path: Path = Path("client_cache/frames")
    cf_access_client_id: Optional[str] = None
    cf_access_client_secret: Optional[str] = None


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _load_yaml(base_path: Path) -> dict:
    """
    Load configuration from dot.yaml.
    Falls back to environment variables for overrides.
    """
    config: dict = {}
    
    yaml_file = base_path / "dot.yaml"
    if yaml_file.exists():
        with open(yaml_file) as f:
            config = yaml.safe_load(f) or {}
    
    # Overlay os.environ for runtime overrides
    import os
    env_overrides = {k: v for k, v in os.environ.items() if v is not None}
    
    return config, env_overrides


def load_config(
    role: Literal["server", "client", "shared"] = "shared",
) -> SharedConfig:
    """
    Build the typed configuration for the requested role.
    """

    base_path = Path(".").resolve()
    config, env = _load_yaml(base_path)
    
    # Helper to get nested yaml value or env override
    def get(yaml_path: str, env_key: str, default=None):
        # Check env override first
        if env_key in env:
            return env[env_key]
        # Navigate yaml path (e.g., "mdai.domain" -> config["mdai"]["domain"])
        parts = yaml_path.split(".")
        val = config
        for part in parts:
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                return default
        return val if val is not None else default
    
    # Required fields
    data_dir = get("server.data_dir", "DATA_DIR", "data")
    domain = get("mdai.domain", "DOMAIN")
    project_id = get("mdai.project_id", "PROJECT_ID")
    dataset_id = get("mdai.dataset", "DATASET_ID")
    label_id = get("mdai.label_id", "LABEL_ID")
    empty_id = get("mdai.empty_id", "EMPTY_ID")
    
    required = {
        "domain": domain,
        "project_id": project_id,
        "dataset_id": dataset_id,
        "label_id": label_id,
        "empty_id": empty_id,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ConfigError(f"Missing required config keys: {', '.join(missing)}")

    # MDAI_TOKEN is optional - can be set later via /api/settings
    mdai_token = get("mdai.token", "MDAI_TOKEN", "not_configured_yet")

    shared_kwargs = dict(
        mdai_token=mdai_token,
        data_dir=Path(data_dir).expanduser().resolve(),
        domain=domain,
        project_id=project_id,
        dataset_id=dataset_id,
        label_id=label_id,
        empty_id=empty_id,
        flow_method=get("optical_flow.method", "FLOW_METHOD", "dis"),
        test_study_uid=get("test.study_uid", "TEST_STUDY_UID"),
        test_series_uid=get("test.series_uid", "TEST_SERIES_UID"),
    )

    if role == "server":
        server_kwargs = dict(
            server_host=get("server.host", "SERVER_HOST", "0.0.0.0"),
            server_port=int(get("server.port", "SERVER_PORT", 5000)),
            server_state_path=(
                base_path / get("server.state_path", "SERVER_STATE_PATH", "server_state")
            ).resolve(),
            mask_storage_path=(
                base_path / get("server.mask_dir", "MASK_STORAGE_PATH", "output")
            ).resolve(),
            recent_view_threshold_minutes=int(
                get("server.recent_minutes", "RECENT_VIEW_THRESHOLD_MINUTES", 60)
            ),
            retrack_workers=int(get("server.retrack_workers", "RETRACK_WORKERS", 2)),
        )
        return ServerConfig(**shared_kwargs, **server_kwargs)

    if role == "client":
        # Get Cloudflare headers
        cf_headers = config.get("cloudflare_headers", {}) or {}
        cf_id = cf_headers.get("CF-Access-Client-Id") or env.get("CF_ACCESS_CLIENT_ID")
        cf_secret = cf_headers.get("CF-Access-Client-Secret") or env.get("CF_ACCESS_CLIENT_SECRET")
        
        client_kwargs = dict(
            client_port=int(get("client.port", "CLIENT_PORT", 8080)),
            server_url=get("server.url", "SERVER_URL", "http://localhost:5000"),
            user_email=get("client.user_email", "USER_EMAIL"),
            video_cache_path=Path(
                get("server.video_cache", "VIDEO_CACHE_PATH", "client_cache/data")
            ).resolve(),
            frames_path=Path(
                get("server.frame_cache", "FRAMES_CACHE_PATH", "client_cache/frames")
            ).resolve(),
            cf_access_client_id=cf_id,
            cf_access_client_secret=cf_secret,
        )
        return ClientConfig(**shared_kwargs, **client_kwargs)

    return SharedConfig(**shared_kwargs)
