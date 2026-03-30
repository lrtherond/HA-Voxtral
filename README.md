# Wyoming Voxtral

Wyoming protocol server for Mistral's Voxtral TTS API.

This project exposes Voxtral TTS as a [Wyoming](https://github.com/OHF-Voice/wyoming) server so Wyoming clients such as Home Assistant can use Mistral voices through a local TCP endpoint.

## What It Does

- Exposes Voxtral TTS over the Wyoming protocol
- Auto-discovers saved Mistral voices from your account
- Supports local reference-audio voices for zero-shot cloning
- Supports Wyoming streaming synthesis with sentence-aware chunking
- Runs cleanly with `uvx` or `uv run`

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- A Mistral API key
- At least one available voice:
  - a saved voice already created in Mistral, or
  - one or more local reference audio files

## Configuration

The server reads configuration from a local `.env` file automatically.

Create `.env`:

```dotenv
MISTRAL_API_KEY=your-mistral-api-key
MISTRAL_API_URL=https://api.mistral.ai/v1
VOXTRAL_MODEL=voxtral-mini-tts-2603
WYOMING_URI=tcp://0.0.0.0:10300
WYOMING_LANGUAGES=en
```

You can copy from [`.env.example`](.env.example).

## Install And Run

### Run Directly With `uvx`

From the project root:

```bash
uvx --from . wyoming-voxtral
```

Or with explicit options:

```bash
uvx --from . wyoming-voxtral \
  --uri tcp://0.0.0.0:10300 \
  --languages en fr
```

If you want to expose local reference voices:

```bash
uvx --from . wyoming-voxtral \
  --reference-voice-dir ./voices
```

### Run In Development Mode

Install the project and dev tooling:

```bash
uv sync
```

Run the server:

```bash
uv run wyoming-voxtral
```

## Voice Sources

The server can expose two kinds of voices.

### Saved Mistral Voices

If saved-voice discovery is enabled, the server loads voices from Mistral at startup using the authenticated API key.

This is the default behavior.

### Local Reference Voices

You can point the server at a directory of audio files:

```bash
uvx --from . wyoming-voxtral --reference-voice-dir ./voices
```

Supported file types:

- `.aac`
- `.flac`
- `.m4a`
- `.mp3`
- `.ogg`
- `.opus`
- `.wav`

The filename stem becomes the Wyoming voice name.

Example:

- `voices/alice.wav` becomes `alice`
- `voices/news_reader.mp3` becomes `news_reader`

## Common Commands

Run with saved voices only:

```bash
uvx --from . wyoming-voxtral
```

Run with local reference voices only:

```bash
uvx --from . wyoming-voxtral \
  --disable-saved-voice-discovery \
  --reference-voice-dir ./voices
```

Run on a custom port:

```bash
uvx --from . wyoming-voxtral \
  --uri tcp://0.0.0.0:10301
```

Tune streaming chunking:

```bash
uvx --from . wyoming-voxtral \
  --streaming-min-words 8 \
  --streaming-max-chars 180
```

## Important CLI Options

- `--uri`
  Wyoming server URI. Default: `tcp://0.0.0.0:10300`
- `--languages`
  Languages advertised for configured voices
- `--mistral-api-key`
  Overrides `MISTRAL_API_KEY`
- `--mistral-api-url`
  Overrides `MISTRAL_API_URL`
- `--tts-model`
  Voxtral model to use. Default: `voxtral-mini-tts-2603`
- `--reference-voice-dir`
  Directory of local reference clips
- `--disable-saved-voice-discovery`
  Skip loading saved voices from Mistral
- `--sample-rate`
  Wyoming audio sample rate. Default: `24000`
- `--streaming-min-words`
  Optional minimum words per synthesis chunk
- `--streaming-max-chars`
  Optional maximum characters per synthesis chunk
- `--request-timeout`
  HTTP timeout for Mistral requests

## Language Support

Sentence segmentation uses [pySBD](https://github.com/nipunsadvilkar/pySBD). If the requested language is not supported by pySBD, the server falls back to English segmentation automatically and logs a warning. Synthesis itself is not affected — Voxtral handles the language independently.

Supported pySBD languages include: `en`, `de`, `es`, `fr`, `it`, `ja`, `nl`, `pl`, `pt`, `ru`, `tr`, `zh`.

## How Audio Is Handled

- Voxtral streaming is requested in `pcm`
- Voxtral's float32 PCM is converted to PCM16 for Wyoming compatibility
- The server advertises streaming TTS support to Wyoming clients
- Long streamed text is chunked on sentence boundaries for lower perceived latency

## Home Assistant Notes

This server is intended to sit behind Home Assistant's Wyoming integration.

Typical flow:

1. Start the server locally or on another reachable machine.
2. Point Home Assistant's Wyoming TTS integration at the server host and port.
3. Select one of the discovered voices.

## Development

This repository is structured around `uv` with:

- `black`
- `ruff`
- `ty`
- `bandit`
- `pytest`

Install everything:

```bash
uv sync
```

Run the checks:

```bash
uv run black --check .
uv run ruff check .
uv run ty check src tests
uv run bandit -c pyproject.toml -r src
uv run pytest
```

## Current Status

The server has been validated against the live Voxtral API for:

- saved voice discovery
- saved-voice streaming synthesis
- saved voice sample retrieval
- `ref_audio` streaming synthesis

It is intended to be run with `uvx`, not Docker.
