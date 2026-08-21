"""Shared utility helpers for randomness control, YAML loading, and filesystem setup."""

import os
import random
from pathlib import Path

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_config_path(path: str | Path, extra_search_dirs: list[str | Path] | None = None) -> Path:
    """Resolve a config path, supporting bare filenames via common config folders."""
    candidate = Path(path)
    if candidate.exists():
        return candidate

    # If the user passed just a filename (no directory), search common config folders.
    if candidate.parent == Path("."):
        search_dirs: list[Path] = [
            Path("configs/train"),
            Path("configs/data"),
            Path("configs/features"),
            Path("configs/models"),
            Path("configs/llm"),
            Path("configs/prompts"),
        ]
        if extra_search_dirs:
            search_dirs.extend(Path(d) for d in extra_search_dirs)

        for directory in search_dirs:
            resolved = directory / candidate.name
            if resolved.exists():
                return resolved

    return candidate


def load_yaml(path: str | Path) -> dict:
    resolved_path = resolve_config_path(path)
    if not resolved_path.exists():
        cwd = Path.cwd()
        raise FileNotFoundError(
            f"YAML config not found: {path}. CWD: {cwd}. "
            "Tip: pass a full relative path like configs/train/<name>.yaml"
        )

    with open(resolved_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
