"""Thin async client for the Voxtral TTS API."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

import httpx

from .const import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_MISTRAL_BASE_URL,
    DEFAULT_POOL_TIMEOUT,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_RETRY_MAX_DELAY,
    __version__,
)
from .models import SavedVoice
from .utilities import make_unique_name, pcm_f32le_to_s16le

_LOGGER = logging.getLogger(__name__)


class MistralApiError(RuntimeError):
    """Raised when the Mistral API responds with an unusable result."""


type SleepFunc = Callable[[float], Awaitable[None]]


class TtsStreamClient(Protocol):
    """Structural type for a client that can stream synthesized speech."""

    def stream_speech(
        self,
        *,
        model: str,
        text: str,
        voice_id: str | None = None,
        reference_audio_b64: str | None = None,
    ) -> AsyncIterator[bytes]: ...


class MistralTtsClient:
    """Minimal client for saved-voice discovery and speech streaming."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_MISTRAL_BASE_URL,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_RETRY_ATTEMPTS,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
        retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep_func: SleepFunc = asyncio.sleep,
    ) -> None:
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
        self._sleep = sleep_func
        self._client = httpx.AsyncClient(
            base_url=f"{base_url.rstrip('/')}/",
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": f"wyoming-voxtral/{__version__}",
            },
            timeout=httpx.Timeout(
                connect=DEFAULT_CONNECT_TIMEOUT,
                read=timeout,
                write=timeout,
                pool=DEFAULT_POOL_TIMEOUT,
            ),
            transport=transport,
        )

    async def __aenter__(self) -> MistralTtsClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the shared HTTP client."""
        await self._client.aclose()

    async def list_saved_voices(self, default_languages: list[str]) -> list[SavedVoice]:
        """Fetch all saved Mistral voices using offset pagination."""
        discovered_voices: list[SavedVoice] = []
        used_names: set[str] = set()
        offset = 0
        limit = 100

        while True:
            response = await self._request_with_retries(
                "GET",
                "audio/voices",
                action="list voices",
                params={"limit": limit, "offset": offset},
            )

            payload = response.json()
            items = payload.get("items", [])
            if not isinstance(items, list):
                raise MistralApiError("Unexpected voices response: 'items' was not a list")

            total_value = payload.get("total", len(items))
            total = total_value if isinstance(total_value, int) else len(items)

            for item in items:
                if not isinstance(item, dict):
                    continue

                voice_id = item.get("id")
                if not isinstance(voice_id, str) or not voice_id:
                    continue

                raw_name_value = item.get("name") or item.get("slug") or voice_id
                raw_name = raw_name_value if isinstance(raw_name_value, str) else voice_id
                display_name = make_unique_name(raw_name, used_names, fallback="Voice")

                raw_languages = item.get("languages")
                if isinstance(raw_languages, list):
                    languages = tuple(
                        language for language in raw_languages if isinstance(language, str)
                    )
                else:
                    languages = ()

                if not languages:
                    languages = tuple(default_languages)

                gender = item.get("gender")
                gender_suffix = f" ({gender})" if isinstance(gender, str) and gender else ""
                description = f"{raw_name}{gender_suffix}"
                slug = item.get("slug") if isinstance(item.get("slug"), str) else None

                discovered_voices.append(
                    SavedVoice(
                        voice_id=voice_id,
                        display_name=display_name,
                        description=description,
                        languages=languages,
                        raw_name=raw_name,
                        slug=slug,
                    )
                )

            offset += len(items)
            if not items or offset >= total:
                break

        return discovered_voices

    async def get_voice_sample(self, voice_id: str) -> bytes:
        """Fetch the sample audio for a saved voice."""
        response = await self._request_with_retries(
            "GET",
            f"audio/voices/{voice_id}/sample",
            action="get voice sample",
        )
        return response.content

    async def stream_speech(
        self,
        *,
        model: str,
        text: str,
        voice_id: str | None = None,
        reference_audio_b64: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield PCM16 audio chunks from Voxtral's streaming speech API."""
        if (voice_id is None) == (reference_audio_b64 is None):
            raise ValueError("Exactly one of voice_id or reference_audio_b64 must be provided")

        payload: dict[str, Any] = {
            "model": model,
            "input": text,
            "response_format": "pcm",
            "stream": True,
        }
        if voice_id is not None:
            payload["voice_id"] = voice_id
        else:
            payload["ref_audio"] = reference_audio_b64

        headers = {"Accept": "text/event-stream"}
        attempt = 0
        while True:
            yielded_audio = False
            try:
                async with self._client.stream(
                    "POST",
                    "audio/speech",
                    json=payload,
                    headers=headers,
                ) as response:
                    if (
                        self._is_retriable_status(response.status_code)
                        and attempt < self._max_retries
                    ):
                        delay = self._get_retry_delay(response, attempt)
                        _LOGGER.warning(
                            "Retrying stream speech after HTTP %d in %.2fs",
                            response.status_code,
                            delay,
                        )
                        await response.aread()
                        await self._sleep(delay)
                        attempt += 1
                        continue

                    await self._raise_for_status(response, action="stream speech")

                    async for event_name, event_data in self._iter_sse_events(response):
                        if event_name == "speech.audio.delta":
                            audio_data = event_data.get("audio_data")
                            if not isinstance(audio_data, str):
                                raise MistralApiError(
                                    "Streaming speech chunk did not include audio_data"
                                )

                            yielded_audio = True
                            yield pcm_f32le_to_s16le(base64.b64decode(audio_data))
                        elif event_name == "speech.audio.done":
                            _LOGGER.debug("Completed streaming speech request for model %s", model)
                            return
                        elif event_name == "error":
                            raise MistralApiError(
                                f"Streaming speech returned an error event: {event_data!r}"
                            )

                return
            except (httpx.NetworkError, httpx.TimeoutException) as exc:
                if yielded_audio or attempt >= self._max_retries:
                    raise MistralApiError(
                        f"Failed to stream speech: {self._describe_exception(exc)}"
                    ) from exc

                delay = self._get_retry_delay(None, attempt)
                _LOGGER.warning(
                    "Retrying stream speech after transport error in %.2fs: %s",
                    delay,
                    self._describe_exception(exc),
                )
                await self._sleep(delay)
                attempt += 1

    async def _iter_sse_events(
        self, response: httpx.Response
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Parse a text/event-stream response."""
        event_name = ""
        data_lines: list[str] = []

        async for line in response.aiter_lines():
            if not line:
                if event_name and data_lines:
                    yield event_name, self._decode_sse_data(event_name, data_lines)
                event_name = ""
                data_lines = []
                continue

            if line.startswith(":"):
                continue

            field_name, _, field_value = line.partition(":")
            field_value = field_value.lstrip()

            if field_name == "event":
                event_name = field_value
            elif field_name == "data":
                data_lines.append(field_value)

        if event_name and data_lines:
            yield event_name, self._decode_sse_data(event_name, data_lines)

    def _decode_sse_data(self, event_name: str, data_lines: list[str]) -> dict[str, Any]:
        """Decode the JSON payload for a single SSE event."""
        payload = "\n".join(data_lines)
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise MistralApiError(
                f"Invalid JSON payload for SSE event {event_name!r}: {payload}"
            ) from exc

        if isinstance(decoded, dict):
            return decoded

        raise MistralApiError(
            f"Unexpected non-object payload for SSE event {event_name!r}: {decoded!r}"
        )

    async def _request_with_retries(
        self,
        method: str,
        url: str,
        *,
        action: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Send a request with bounded retries for safe retry conditions."""
        attempt = 0
        while True:
            try:
                response = await self._client.request(method, url, **kwargs)
            except (httpx.NetworkError, httpx.TimeoutException) as exc:
                if attempt >= self._max_retries:
                    raise MistralApiError(
                        f"Failed to {action}: {self._describe_exception(exc)}"
                    ) from exc

                delay = self._get_retry_delay(None, attempt)
                _LOGGER.warning(
                    "Retrying %s after transport error in %.2fs: %s",
                    action,
                    delay,
                    self._describe_exception(exc),
                )
                await self._sleep(delay)
                attempt += 1
                continue

            if self._is_retriable_status(response.status_code) and attempt < self._max_retries:
                delay = self._get_retry_delay(response, attempt)
                _LOGGER.warning(
                    "Retrying %s after HTTP %d in %.2fs",
                    action,
                    response.status_code,
                    delay,
                )
                await response.aread()
                await self._sleep(delay)
                attempt += 1
                continue

            await self._raise_for_status(response, action=action)
            return response

    def _is_retriable_status(self, status_code: int) -> bool:
        """Return True when an HTTP status is safe to retry."""
        return status_code in {429, 500, 502, 503, 504}

    def _get_retry_delay(self, response: httpx.Response | None, attempt: int) -> float:
        """Resolve retry delay, preferring Retry-After when present."""
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            parsed_retry_after = self._parse_retry_after(retry_after)
            if parsed_retry_after is not None:
                return min(parsed_retry_after, self._retry_max_delay)

        delay = self._retry_base_delay * (2**attempt)
        return min(delay, self._retry_max_delay)

    def _parse_retry_after(self, retry_after: str | None) -> float | None:
        """Parse Retry-After header values as seconds."""
        if not retry_after:
            return None

        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            pass

        try:
            retry_at = parsedate_to_datetime(retry_after)
        except (TypeError, ValueError):
            return None

        now = datetime.now(UTC)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max((retry_at - now).total_seconds(), 0.0)

    def _describe_exception(self, exc: Exception) -> str:
        """Return a concise message for transport exceptions."""
        if str(exc):
            return str(exc)
        return exc.__class__.__name__

    async def _raise_for_status(self, response: httpx.Response, *, action: str) -> None:
        """Convert HTTP client errors into friendlier runtime errors."""
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = await self._extract_error_detail(exc.response)
            if detail:
                raise MistralApiError(f"Failed to {action}: {detail}") from exc
            raise MistralApiError(f"Failed to {action}: HTTP {exc.response.status_code}") from exc

    async def _extract_error_detail(self, response: httpx.Response) -> str:
        """Extract a concise error detail from a failed HTTP response."""
        try:
            payload = await response.aread()
        except Exception:  # pragma: no cover - best-effort error reporting
            return ""

        text = payload.decode("utf-8", errors="replace").strip()
        if not text:
            return ""

        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return text

        if isinstance(decoded, dict):
            if isinstance(decoded.get("message"), str):
                return decoded["message"]
            error = decoded.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                return error["message"]
            if isinstance(error, str):
                return error
            if isinstance(decoded.get("detail"), str):
                return decoded["detail"]

        return text
