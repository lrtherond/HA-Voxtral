from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

import wyoming_voxtral.__main__ as main_module
from wyoming_voxtral.__main__ import main
from wyoming_voxtral.models import ReferenceVoice


class FakeClient:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    async def __aenter__(self) -> FakeClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def close(self) -> None:
        return None

    async def list_saved_voices(self, default_languages):
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
