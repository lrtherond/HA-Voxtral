"""CLI entry point for the Wyoming Voxtral server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import warnings
from functools import partial
from pathlib import Path

import httpx
from wyoming.server import AsyncServer

from .catalog import create_info, create_tts_voices
from .client import MistralApiError, MistralTtsClient
from .const import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_MISTRAL_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_URI,
    __version__,
)
from .utilities import configure_logging, env_flag, load_dotenv, load_reference_voices

warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"pysbd(\.|$)",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--uri", default=os.getenv("WYOMING_URI", DEFAULT_URI), help="Wyoming server URI"
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("WYOMING_LOG_LEVEL", DEFAULT_LOG_LEVEL),
        help="Logging level",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=os.getenv("WYOMING_LANGUAGES", "en").split(),
        help="Languages supported by the configured voices",
    )
    parser.add_argument(
        "--mistral-api-key",
        default=os.getenv("MISTRAL_API_KEY"),
        required=not os.getenv("MISTRAL_API_KEY"),
        help="Mistral API key",
    )
    parser.add_argument(
        "--mistral-api-url",
        default=os.getenv("MISTRAL_API_URL", DEFAULT_MISTRAL_BASE_URL),
        help="Base URL for the Mistral API",
    )
    parser.add_argument(
        "--tts-model",
        default=os.getenv("VOXTRAL_MODEL", DEFAULT_MODEL),
        help="Voxtral TTS model identifier",
    )
    parser.add_argument(
        "--reference-voice-dir",
        type=Path,
        default=(
            Path(value).expanduser()
            if (value := os.getenv("VOXTRAL_REFERENCE_VOICE_DIR"))
            else None
        ),
        help="Directory with local reference audio clips to expose as Wyoming voices",
    )
    parser.add_argument(
        "--disable-saved-voice-discovery",
        action="store_true",
        default=env_flag("VOXTRAL_DISABLE_SAVED_VOICE_DISCOVERY", default=False),
        help="Do not call the saved-voice listing API at startup",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=int(os.getenv("VOXTRAL_SAMPLE_RATE", str(DEFAULT_SAMPLE_RATE))),
        help="Sample rate reported to Wyoming after PCM conversion",
    )
    parser.add_argument(
        "--streaming-min-words",
        type=int,
        default=int(value) if (value := os.getenv("VOXTRAL_STREAMING_MIN_WORDS")) else None,
        help="Optional minimum word count per streaming chunk",
    )
    parser.add_argument(
        "--streaming-max-chars",
        type=int,
        default=int(value) if (value := os.getenv("VOXTRAL_STREAMING_MAX_CHARS")) else None,
        help="Optional maximum characters per streaming chunk",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=float(os.getenv("MISTRAL_TIMEOUT", "60")),
        help="HTTP timeout in seconds for Mistral requests",
    )

    return parser


async def main() -> None:
    """Run the Wyoming Voxtral server."""
    from .handler import VoxtralEventHandler

    try:
        load_dotenv()
    except ValueError as exc:
        raise SystemExit(f"Invalid .env file: {exc}") from exc

    parser = build_parser()
    args = parser.parse_args()

    if args.sample_rate <= 0:
        parser.error("--sample-rate must be a positive integer")

    if args.streaming_min_words is not None and args.streaming_min_words <= 0:
        parser.error("--streaming-min-words must be a positive integer")

    if args.streaming_max_chars is not None and args.streaming_max_chars <= 0:
        parser.error("--streaming-max-chars must be a positive integer")

    try:
        configure_logging(args.log_level)
    except ValueError as exc:
        parser.error(str(exc))

    logger = logging.getLogger(__name__)
    logger.info("Starting Wyoming Voxtral %s", __version__)

    try:
        reference_voices = load_reference_voices(args.reference_voice_dir, args.languages)
    except ValueError as exc:
        parser.error(str(exc))

    if reference_voices:
        logger.info(
            "Loaded %d reference voice(s) from %s", len(reference_voices), args.reference_voice_dir
        )

    client = MistralTtsClient(
        api_key=args.mistral_api_key,
        base_url=args.mistral_api_url,
        timeout=args.request_timeout,
    )

    async with client:
        saved_voices = []
        if not args.disable_saved_voice_discovery:
            try:
                saved_voices = await client.list_saved_voices(args.languages)
            except (MistralApiError, httpx.HTTPError, httpx.TimeoutException) as exc:
                if reference_voices:
                    logger.warning(
                        (
                            "Saved voice discovery failed; continuing with local "
                            "reference voices only: %s"
                        ),
                        exc,
                    )
                else:
                    parser.error(
                        "Failed to discover saved voices from Mistral and no local "
                        f"reference voices were loaded: {exc}"
                    )

        voices = create_tts_voices(
            model_name=args.tts_model,
            saved_voices=saved_voices,
            reference_voices=reference_voices,
        )
        if not voices:
            parser.error(
                "No voices available. Create saved voices in Mistral or provide "
                "--reference-voice-dir."
            )

        logger.info("Configured %d Wyoming voice(s)", len(voices))
        info = create_info(voices)
        server = AsyncServer.from_uri(args.uri)

        logger.info("Starting server at %s", args.uri)
        await server.run(
            partial(
                VoxtralEventHandler,
                info=info,
                tts_client=client,
                sample_rate=args.sample_rate,
                tts_streaming_min_words=args.streaming_min_words,
                tts_streaming_max_chars=args.streaming_max_chars,
            )
        )


def run() -> None:
    """Synchronous console script wrapper."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
