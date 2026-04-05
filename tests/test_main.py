from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

import wyoming_voxtral.__main__ as main_module
from wyoming_voxtral.__main__ import main
from wyoming_voxtral.client import MistralApiError
from wyoming_voxtral.models import ReferenceVoice


class FakeClient:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self._discovery_error: Exception | None = None

    async def __aenter__(self) -> FakeClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def close(self) -> None:
        return None

    async def list_saved_voices(self, default_languages):
        if self._discovery_error:
            raise self._discovery_error
        return []


class CapturingServer:
    def __init__(self) -> None:
        self.handler = None

    async def run(self, handler_factory) -> None:
        self.handler = handler_factory(Mock(name="reader"), Mock(name="writer"))


@pytest.mark.asyncio
async def test_main_rejects_empty_voice_catalog(monkeypatch, capsys):
    monkeypatch.setattr(main_module, "MistralTtsClient", FakeClient)
    monkeypatch.setattr(main_module, "load_reference_voices", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(sys, "argv", ["wyoming-voxtral", "--mistral-api-key", "test-key"])

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "No voices available" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_falls_back_to_reference_voices_when_discovery_raises_mistral_api_error(
    monkeypatch,
):
    """A MistralApiError (including wrapped JSON parse failures) must not abort startup
    when local reference voices are available."""
    reference_voice = ReferenceVoice(
        display_name="Alice",
        description="Reference voice from Alice.wav",
        languages=("en",),
        source_path=Path("/tmp/Alice.wav"),
        reference_audio_b64="c2FtcGxl",
    )
    server = CapturingServer()

    failing_client = FakeClient()
    failing_client._discovery_error = MistralApiError("Failed to parse voices response: bad json")

    monkeypatch.setattr(main_module, "MistralTtsClient", lambda *a, **kw: failing_client)
    monkeypatch.setattr(
        main_module, "load_reference_voices", lambda *_args, **_kwargs: [reference_voice]
    )
    monkeypatch.setattr(main_module.AsyncServer, "from_uri", staticmethod(lambda uri: server))
    monkeypatch.setattr(
        sys,
        "argv",
        ["wyoming-voxtral", "--mistral-api-key", "test-key"],
    )

    await main()  # must not raise

    assert server.handler is not None


@pytest.mark.parametrize(
    ("argv_extra", "expected_fragment"),
    [
        (["--streaming-min-words", "0"], "--streaming-min-words"),
        (["--streaming-min-words", "-1"], "--streaming-min-words"),
        (["--streaming-max-chars", "0"], "--streaming-max-chars"),
        (["--streaming-max-chars", "-5"], "--streaming-max-chars"),
        (["--sample-rate", "0"], "--sample-rate"),
    ],
)
async def test_main_rejects_invalid_numeric_options(
    monkeypatch, capsys, argv_extra, expected_fragment
):
    monkeypatch.setattr(main_module, "MistralTtsClient", FakeClient)
    monkeypatch.setattr(main_module, "load_reference_voices", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        sys,
        "argv",
        ["wyoming-voxtral", "--mistral-api-key", "test-key"] + argv_extra,
    )

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert expected_fragment in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_builds_handler_with_reference_voice(monkeypatch):
    reference_voice = ReferenceVoice(
        display_name="Alice",
        description="Reference voice from Alice.wav",
        languages=("en",),
        source_path=Path("/tmp/Alice.wav"),
        reference_audio_b64="c2FtcGxl",
    )
    server = CapturingServer()

    monkeypatch.setattr(main_module, "MistralTtsClient", FakeClient)
    monkeypatch.setattr(
        main_module, "load_reference_voices", lambda *_args, **_kwargs: [reference_voice]
    )
    monkeypatch.setattr(main_module.AsyncServer, "from_uri", staticmethod(lambda uri: server))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wyoming-voxtral",
            "--mistral-api-key",
            "test-key",
            "--disable-saved-voice-discovery",
        ],
    )

    await main()

    assert server.handler is not None
