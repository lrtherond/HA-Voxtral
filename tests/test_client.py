from __future__ import annotations

import base64
import struct

import httpx
import pytest

from wyoming_voxtral.client import MistralApiError, MistralTtsClient


class AsyncBytesStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


class FailingAsyncBytesStream(httpx.AsyncByteStream):
    def __init__(self, first_chunk: bytes) -> None:
        self._first_chunk = first_chunk

    async def __aiter__(self):
        yield self._first_chunk
        raise httpx.ReadError("boom")

    async def aclose(self) -> None:
        return None


def _make_delta_event(float_samples: bytes) -> bytes:
    audio_data = base64.b64encode(float_samples).decode("ascii")
    payload = f'event: speech.audio.delta\ndata: {{"audio_data":"{audio_data}"}}\n\n'
    return payload.encode("utf-8")


def _make_done_event() -> bytes:
    return b'event: speech.audio.done\ndata: {"usage":{"tokens":1}}\n\n'


@pytest.mark.asyncio
async def test_list_saved_voices_retries_on_rate_limit():
    sleeps: list[float] = []
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"message": "slow down"})
        return httpx.Response(
            200,
            json={
                "items": [{"id": "voice-1", "name": "Alice", "languages": ["en"]}],
                "total": 1,
            },
        )

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    client = MistralTtsClient(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        retry_base_delay=0.01,
        retry_max_delay=0.01,
        sleep_func=fake_sleep,
    )

    async with client:
        voices = await client.list_saved_voices(["en"])

    assert calls == 2
    assert sleeps == [0.01]
    assert voices[0].display_name == "Alice"


@pytest.mark.asyncio
async def test_stream_speech_retries_before_audio_begins():
    sleeps: list[float] = []
    calls = 0
    float_samples = struct.pack("<ff", 0.0, 0.5)

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"message": "temporary"})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=AsyncBytesStream([_make_delta_event(float_samples), _make_done_event()]),
        )

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    client = MistralTtsClient(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        retry_base_delay=0.01,
        retry_max_delay=0.01,
        sleep_func=fake_sleep,
    )

    async with client:
        chunks = [
            chunk
            async for chunk in client.stream_speech(
                model="voxtral-mini-tts-2603",
                text="hello",
                voice_id="voice-1",
            )
        ]

    assert calls == 2
    assert sleeps == [0.01]
    assert chunks == [struct.pack("<hh", 0, 16383)]


@pytest.mark.asyncio
async def test_stream_speech_does_not_retry_after_audio_has_started():
    calls = 0
    float_samples = struct.pack("<ff", 0.0, 0.5)

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=FailingAsyncBytesStream(_make_delta_event(float_samples)),
        )

    client = MistralTtsClient(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        retry_base_delay=0.01,
        retry_max_delay=0.01,
    )

    collected = []
    async with client:
        with pytest.raises(MistralApiError, match="Failed to stream speech"):
            async for chunk in client.stream_speech(
                model="voxtral-mini-tts-2603",
                text="hello",
                voice_id="voice-1",
            ):
                collected.append(chunk)

    assert calls == 1
    assert collected == [struct.pack("<hh", 0, 16383)]


@pytest.mark.asyncio
async def test_stream_speech_raises_on_error_event():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=AsyncBytesStream([b'event: error\ndata: {"message":"bad request"}\n\n']),
        )

    client = MistralTtsClient(api_key="test-key", transport=httpx.MockTransport(handler))

    async with client:
        with pytest.raises(MistralApiError, match="error event"):
            async for _chunk in client.stream_speech(
                model="voxtral-mini-tts-2603",
                text="hello",
                voice_id="voice-1",
            ):
                pass


@pytest.mark.asyncio
async def test_stream_speech_raises_on_invalid_sse_payload():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=AsyncBytesStream([b"event: speech.audio.delta\ndata: not-json\n\n"]),
        )

    client = MistralTtsClient(api_key="test-key", transport=httpx.MockTransport(handler))

    async with client:
        with pytest.raises(MistralApiError, match="Invalid JSON payload"):
            async for _chunk in client.stream_speech(
                model="voxtral-mini-tts-2603",
                text="hello",
                voice_id="voice-1",
            ):
                pass
