"""Utility helpers for logging, PCM conversion, and local voices."""

from __future__ import annotations

import array
import base64
import logging
import math
import os
import sys
from pathlib import Path

from .const import SUPPORTED_REFERENCE_EXTENSIONS
from .models import ReferenceVoice


def configure_logging(level: str) -> None:
    """Configure root logging from a string level."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(level=numeric_level, force=True)


def env_flag(name: str, default: bool = False) -> bool:
    """Parse a common boolean-style environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_dotenv(path: Path | None = None) -> None:
    """Load environment variables from a local `.env` file when present."""
    dotenv_path = (path or Path(".env")).expanduser()
    if not dotenv_path.exists():
        return
    if not dotenv_path.is_file():
        raise ValueError(f".env path is not a file: {dotenv_path}")

    for line_number, raw_line in enumerate(dotenv_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env entry at line {line_number}: {raw_line!r}")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid .env entry at line {line_number}: empty variable name")

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def make_unique_name(preferred: str, used_names: set[str], *, fallback: str) -> str:
    """Return a display name that is unique within the provided set."""
    base_name = preferred.strip() or fallback
    candidate = base_name
    suffix = 2

    while candidate in used_names:
        candidate = f"{base_name} {suffix}"
        suffix += 1

    used_names.add(candidate)
    return candidate


def load_reference_voices(
    directory: Path | None, default_languages: list[str]
) -> list[ReferenceVoice]:
    """Load local reference audio files as Wyoming voices."""
    if directory is None:
        return []

    directory = directory.expanduser().resolve()
    if not directory.exists():
        raise ValueError(f"Reference voice directory does not exist: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Reference voice path is not a directory: {directory}")

    reference_voices: list[ReferenceVoice] = []
    used_names: set[str] = set()

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_REFERENCE_EXTENSIONS:
            continue

        display_name = make_unique_name(path.stem, used_names, fallback="Reference Voice")
        reference_voices.append(
            ReferenceVoice(
                display_name=display_name,
                description=f"Reference voice from {path.name}",
                languages=tuple(default_languages),
                source_path=path,
                reference_audio_b64=base64.b64encode(path.read_bytes()).decode("ascii"),
            )
        )

    return reference_voices


def pcm_f32le_to_s16le(audio_bytes: bytes) -> bytes:
    """Convert float32 little-endian PCM to 16-bit little-endian PCM."""
    if not audio_bytes:
        return b""

    if len(audio_bytes) % 4 != 0:
        raise ValueError("PCM float32 payload length must be a multiple of 4 bytes")

    samples: array.array[float] = array.array("f")
    samples.frombytes(audio_bytes)
    if sys.byteorder == "big":
        samples.byteswap()  # input is little-endian; swap to native for iteration

    def _quantize(s: float) -> int:
        if math.isnan(s) or math.isinf(s):
            return 0
        if s >= 1.0:
            return 32767
        if s <= -1.0:
            return -32768
        return int(s * 32767.0)

    out: array.array[int] = array.array("h", (_quantize(s) for s in samples))
    if sys.byteorder == "big":
        out.byteswap()  # output must be little-endian

    return out.tobytes()
