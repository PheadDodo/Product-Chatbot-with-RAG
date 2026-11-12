"""Validated settings for the existing local Windows model workflow.

Settings are read only when ``load_local_config()`` is called. Supported
environment variables (all optional):

* RAG_MODEL_PATH: GGUF model file; defaults to the existing D: model path.
* RAG_CUDA_BIN_DIR: CUDA DLL folder; defaults to D:/ml_projects/cuda/bin/x64.
* RAG_N_GPU_LAYERS: GPU layer count, including 0 (CPU) or -1 (all); default 52.
* RAG_N_CTX: model context token capacity; default 10000.
* RAG_N_BATCH: logical prompt batch size; default 512.
* RAG_N_UBATCH: physical prompt batch size; default 128.
* RAG_MAX_OUTPUT_TOKENS: completion token allowance; default 1024.
* RAG_RETRIEVAL_TIMEOUT_SECONDS: finite positive timeout; default 180.
* RAG_MAX_QUESTION_CHARS: input character limit; default 8000.

Relative model/CUDA paths resolve against the supplied project directory,
or this module's directory. Tildes are expanded. Loading configuration does
not check whether these paths exist, create directories, change environment
variables or import native model libraries. The model owner checks required
files/directories immediately before loading a model.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LocalModelConfig:
    """Immutable local settings; numeric validation also applies to direct use."""

    model_path: Path = Path(
        "D:/ml_projects/models/Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf"
    )
    cuda_bin_dir: Path = Path("D:/ml_projects/cuda/bin/x64")
    n_gpu_layers: int = 52
    n_ctx: int = 10000
    n_batch: int = 512
    n_ubatch: int = 128
    max_output_tokens: int = 1024
    retrieval_timeout_seconds: float = 180.0
    max_question_chars: int = 8000

    def __post_init__(self):
        for name in ("model_path", "cuda_bin_dir"):
            value = getattr(self, name)
            if not isinstance(value, Path) or any(
                character in str(value) for character in ("\x00", "\n", "\r")
            ):
                raise ValueError(f"{name} must be a valid pathlib.Path.")

        if type(self.n_gpu_layers) is not int or self.n_gpu_layers < -1:
            raise ValueError("n_gpu_layers must be an integer greater than or equal to -1.")

        for name in (
            "n_ctx", "n_batch", "n_ubatch", "max_output_tokens",
            "max_question_chars",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        timeout = self.retrieval_timeout_seconds
        valid_timeout = False
        if not isinstance(timeout, bool) and isinstance(timeout, (int, float)):
            try:
                valid_timeout = math.isfinite(timeout) and timeout > 0
            except OverflowError:
                pass
        if not valid_timeout:
            raise ValueError("retrieval_timeout_seconds must be finite and positive.")
        if self.n_ubatch > self.n_batch:
            raise ValueError("n_ubatch cannot exceed n_batch.")
        if self.max_output_tokens >= self.n_ctx:
            raise ValueError("max_output_tokens must be smaller than n_ctx.")


_MISSING = object()


def _environment_value(environ, name):
    value = environ.get(name, _MISSING)
    if value is _MISSING:
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value.strip()


def _integer_setting(environ, name, default):
    value = _environment_value(environ, name)
    if value is _MISSING:
        return default
    if re.fullmatch(r"[+-]?[0-9]+", value) is None:
        raise ValueError(f"{name} must be an integer.")
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error


def _timeout_setting(environ, name, default):
    value = _environment_value(environ, name)
    if value is _MISSING:
        return default
    try:
        return float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and positive.") from error


def _path_setting(environ, name, default, project_dir):
    value = _environment_value(environ, name)
    value = str(default) if value is _MISSING else value
    if any(character in value for character in ("\x00", "\n", "\r")):
        raise ValueError(f"{name} must be a valid path.")
    try:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = project_dir / path
        return path.resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"{name} must be a valid path.") from error


def load_local_config(environ=None, project_dir=None):
    """Read and validate known settings without checking model availability.

    ``environ`` is an optional mapping, useful for isolated callers/tests.
    Only the documented RAG_* keys are read; other settings and secrets are
    neither copied nor inspected. The supplied mapping remains unchanged.
    """
    environment = os.environ if environ is None else environ
    if not isinstance(environment, Mapping):
        raise ValueError("environ must be a mapping of environment settings.")
    if isinstance(project_dir, str) and not project_dir.strip():
        raise ValueError("project_dir must be a nonempty path.")
    if isinstance(project_dir, (str, Path)) and any(
        character in str(project_dir) for character in ("\x00", "\n", "\r")
    ):
        raise ValueError("project_dir must be a valid path.")
    try:
        base = (
            Path(__file__).parent if project_dir is None else Path(project_dir)
        ).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("project_dir must be a valid path.") from error

    defaults = LocalModelConfig()
    return LocalModelConfig(
        model_path=_path_setting(
            environment, "RAG_MODEL_PATH", defaults.model_path, base,
        ),
        cuda_bin_dir=_path_setting(
            environment, "RAG_CUDA_BIN_DIR", defaults.cuda_bin_dir, base,
        ),
        n_gpu_layers=_integer_setting(
            environment, "RAG_N_GPU_LAYERS", defaults.n_gpu_layers,
        ),
        n_ctx=_integer_setting(environment, "RAG_N_CTX", defaults.n_ctx),
        n_batch=_integer_setting(environment, "RAG_N_BATCH", defaults.n_batch),
        n_ubatch=_integer_setting(
            environment, "RAG_N_UBATCH", defaults.n_ubatch,
        ),
        max_output_tokens=_integer_setting(
            environment, "RAG_MAX_OUTPUT_TOKENS", defaults.max_output_tokens,
        ),
        retrieval_timeout_seconds=_timeout_setting(
            environment, "RAG_RETRIEVAL_TIMEOUT_SECONDS",
            defaults.retrieval_timeout_seconds,
        ),
        max_question_chars=_integer_setting(
            environment, "RAG_MAX_QUESTION_CHARS", defaults.max_question_chars,
        ),
    )
