from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeVoice,
)

from wyoming_voxtral.catalog import create_info, create_tts_voices
from wyoming_voxtral.handler import VoxtralEventHandler
from wyoming_voxtral.models import ReferenceVoice


class DummyTtsClient:
    def __init__(self, chunks: list[bytes]) -> None:
        self.calls: list[dict[str, object]] = []
        self._chunks = chunks

    async def stream_speech(self, **kwargs):
        self.calls.append(kwargs)
        for chunk in self._chunks:
            yield chunk


class CancelAwareTtsClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.cancelled = False

    async def stream_speech(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise RuntimeError("first request failed")

        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled = True
            raise

        if False:  # pragma: no cover - keeps this as an async generator for typing/runtime
            yield b""


class HangingTtsClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.cancelled = False

    async def stream_speech(self, **kwargs):
        self.calls.append(kwargs)
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled = True
            raise

        if False:  # pragma: no cover - keeps this as an async generator for typing/runtime
            yield b""


class RecordingHandler(VoxtralEventHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[Event] = []

    async def write_event(self, event: Event) -> None:
        self.events.append(event)


def _build_info() -> tuple[ReferenceVoice, Info]:
    reference_voice = ReferenceVoice(
        display_name="Alice",
        description="Reference voice from Alice.wav",
        languages=("en",),
        source_path=Path("/tmp/Alice.wav"),
        reference_audio_b64="c2FtcGxl",
    )
    voices = create_tts_voices(
        model_name="voxtral-mini-tts-2603", saved_voices=[], reference_voices=[reference_voice]
    )
    return reference_voice, create_info(voices)


@pytest.mark.asyncio
async def test_handle_synthesize_streams_reference_voice():
    _, info = _build_info()
    client = DummyTtsClient([b"\x01\x00\x02\x00", b"\x03\x00\x04\x00"])
    handler = RecordingHandler(
        MagicMock(name="reader"),
        MagicMock(name="writer"),
        info=info,
        tts_client=client,
        sample_rate=24000,
    )

    event = Synthesize(
        text="Hello world", voice=SynthesizeVoice(name="Alice", language="en")
    ).event()
    assert await handler.handle_event(event) is True

    events = handler.events
    assert [event.type for event in events] == [
        "audio-start",
        "audio-chunk",
        "audio-chunk",
        "audio-stop",
    ]
    assert client.calls[0]["reference_audio_b64"] == "c2FtcGxl"


@pytest.mark.asyncio
async def test_streaming_synthesis_sends_stop_event():
    _, info = _build_info()
    client = DummyTtsClient([b"\x01\x00\x02\x00"])
    handler = RecordingHandler(
        MagicMock(name="reader"),
        MagicMock(name="writer"),
        info=info,
        tts_client=client,
        sample_rate=24000,
    )

    assert await handler.handle_event(
        SynthesizeStart(voice=SynthesizeVoice(name="Alice", language="en")).event()
    )
    assert await handler.handle_event(SynthesizeChunk(text="Hello world. ").event())
    assert await handler.handle_event(SynthesizeChunk(text="Second sentence.").event())
    assert await handler.handle_event(SynthesizeStop().event())

    events = handler.events
    assert events[-2].type == "audio-stop"
    assert events[-1].type == "synthesize-stopped"


@pytest.mark.asyncio
async def test_abort_synthesis_cancels_other_in_flight_tasks():
    _, info = _build_info()
    client = CancelAwareTtsClient()
    handler = RecordingHandler(
        MagicMock(name="reader"),
        MagicMock(name="writer"),
        info=info,
        tts_client=client,
        sample_rate=24000,
    )

    assert await handler.handle_event(
        SynthesizeStart(voice=SynthesizeVoice(name="Alice", language="en")).event()
    )

    result = await handler._process_ready_sentences(["First sentence.", "Second sentence."])

    assert result is False
    assert client.cancelled is True


@pytest.mark.asyncio
async def test_handle_describe_returns_streaming_info():
    _, info = _build_info()
    handler = RecordingHandler(
        MagicMock(name="reader"),
        MagicMock(name="writer"),
        info=info,
        tts_client=DummyTtsClient([]),
        sample_rate=24000,
    )

    assert await handler.handle_event(Describe().event()) is True

    assert len(handler.events) == 1
    emitted_info = Info.from_event(handler.events[0])
    assert emitted_info.tts[0].supports_synthesize_streaming is True
    assert emitted_info.tts[0].voices[0].name == "Alice"


@pytest.mark.asyncio
async def test_handler_stop_cancels_inflight_one_shot_synthesis():
    _, info = _build_info()
    client = HangingTtsClient()
    handler = RecordingHandler(
        MagicMock(name="reader"),
        MagicMock(name="writer"),
        info=info,
        tts_client=client,
        sample_rate=24000,
    )

    synthesis_task = asyncio.create_task(
        handler.handle_event(
            Synthesize(
                text="Hello world",
                voice=SynthesizeVoice(name="Alice", language="en"),
            ).event()
        )
    )

    for _ in range(20):
        if client.calls:
            break
        await asyncio.sleep(0)

    await handler.stop()
    result = await synthesis_task

    assert result is False
    assert client.cancelled is True
