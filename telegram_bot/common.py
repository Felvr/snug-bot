from __future__ import annotations

import os
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = ROOT_DIR / ".env"


def load_env_file(env_path: Path | None = None) -> Path:
    env_file = env_path or DEFAULT_ENV_PATH
    if not env_file.exists():
        return env_file

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value
    return env_file


def sanitize_stem(value: str, default: str = "file") -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._-")
    return cleaned or default


def bool_from_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


def int_from_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default
