"""Constants for the Wyoming Voxtral server."""

from __future__ import annotations

import importlib.metadata

PACKAGE_NAME = "wyoming-voxtral"

try:
    __version__ = importlib.metadata.version(PACKAGE_NAME)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.7.0"

ATTRIBUTION_NAME = "Mistral Voxtral TTS"
ATTRIBUTION_URL = "https://docs.mistral.ai/models/voxtral-tts-26-03"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
DEFAULT_MODEL = "voxtral-mini-tts-2603"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_URI = "tcp://0.0.0.0:10300"
DEFAULT_AUDIO_CHANNELS = 1
DEFAULT_AUDIO_WIDTH = 2
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_POOL_TIMEOUT = 10.0
DEFAULT_REQUEST_TIMEOUT = 60.0
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY = 0.5
DEFAULT_RETRY_MAX_DELAY = 8.0
SUPPORTED_REFERENCE_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
}
TTS_CONCURRENT_REQUESTS = 3
TTS_EMPTY_AUDIO_RETRY_ATTEMPTS = 1
