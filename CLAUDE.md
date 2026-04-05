# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A Wyoming protocol TTS server that bridges Mistral's Voxtral TTS API to Home Assistant. It exposes Voxtral voices over the Wyoming TCP protocol so HA's Wyoming integration can use them.

## Commands

```bash
# Install everything (including dev tools)
uv sync

# Run the server
uv run wyoming-voxtral

# Run all checks
uv run black --check .
uv run ruff check .
uv run ty check src tests
uv run bandit -c pyproject.toml -r src
uv run pytest

# Run a single test file
uv run pytest tests/test_client.py

# Run a single test
uv run pytest tests/test_handler.py::test_name
```

## Architecture

```text
src/wyoming_voxtral/
  __main__.py   # CLI entry point, argument parsing, server startup
  handler.py    # Wyoming AsyncEventHandler — receives Wyoming events, drives synthesis
  client.py     # MistralTtsClient — HTTP/SSE client for Voxtral API
  catalog.py    # Converts raw voice data into Wyoming Info/TtsVoice structures
  models.py     # Data classes: SavedVoice, ReferenceVoice, VoxtralVoice
  utilities.py  # load_dotenv, pcm_f32le_to_s16le, load_reference_voices
  const.py      # All constants and defaults
```

### Data flow

1. `__main__.py` starts up: loads `.env`, parses args, calls `client.list_saved_voices()` and `load_reference_voices()`, builds `Info` via `catalog.create_info()`, then starts `AsyncServer`.
2. Per connection, `VoxtralEventHandler` is instantiated.
3. Wyoming events arrive: `Synthesize` (one-shot) or `SynthesizeStart` / `SynthesizeChunk` / `SynthesizeStop` (streaming).
4. For streaming, pySBD segments accumulated text into sentences as chunks arrive; complete sentences are dispatched to `MistralTtsClient.stream_speech()`.
5. Voxtral returns float32 PCM over SSE; `pcm_f32le_to_s16le()` converts it before sending `AudioChunk` events back to the Wyoming client.

### Voice types

- **SavedVoice**: identified by `voice_id` from Mistral's `/audio/voices` API.
- **ReferenceVoice**: local audio file, base64-encoded and sent as `ref_audio` in each synthesis request. Filename stem becomes the Wyoming voice name.
- **VoxtralVoice**: subclass of Wyoming `TtsVoice` carrying `model_name`, `voice_kind`, and exactly one of `voice_id` or `reference_audio_b64`.

### Streaming concurrency

`TTS_CONCURRENT_REQUESTS = 3` controls the semaphore in `handler.py`. When multiple sentences are dispatched in parallel (streaming mode), only the "allowed" task streams directly to Wyoming; others buffer their audio and write it in order once it's their turn. This preserves sentence ordering while parallelising network I/O.

## Configuration

Reads from `.env` at startup (see `.env.example`). Every env var has a corresponding CLI flag that takes precedence. Key vars:

| Env var                                 | CLI flag                          | Default                 |
| --------------------------------------- | --------------------------------- | ----------------------- |
| `MISTRAL_API_KEY`                       | `--mistral-api-key`               | required                |
| `WYOMING_URI`                           | `--uri`                           | `tcp://0.0.0.0:10300`   |
| `VOXTRAL_MODEL`                         | `--tts-model`                     | `voxtral-mini-tts-2603` |
| `VOXTRAL_REFERENCE_VOICE_DIR`           | `--reference-voice-dir`           | none                    |
| `VOXTRAL_DISABLE_SAVED_VOICE_DISCOVERY` | `--disable-saved-voice-discovery` | false                   |

## Toolchain

- **Formatter**: `black` (line length 100)
- **Linter**: `ruff` (ruleset: B, C4, E, F, I, N, RET, SIM, UP, W)
- **Type checker**: `ty` (errors on warnings, concise output)
- **Security scanner**: `bandit` (B101 skipped — asserts used in tests)
- **Tests**: `pytest` with `pytest-asyncio` in auto mode

The `clones/` directory contains a reference implementation (`wyoming_openai`) and is excluded from all linting/formatting/security scanning.
