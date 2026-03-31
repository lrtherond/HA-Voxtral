"""Wyoming event handler for Voxtral TTS."""

from __future__ import annotations

import asyncio
import logging
import warnings
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar, cast

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r".*invalid escape sequence.*",
        category=SyntaxWarning,
    )
    import pysbd

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info, TtsVoice
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeVoice,
)

from .client import TtsStreamClient
from .const import (
    DEFAULT_AUDIO_CHANNELS,
    DEFAULT_AUDIO_WIDTH,
    TTS_CONCURRENT_REQUESTS,
    TTS_EMPTY_AUDIO_RETRY_ATTEMPTS,
)
from .models import VoxtralVoice

_LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


def _truncate_for_log(text: str, max_length: int = 100) -> str:
    """Truncate text for concise logs."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


@dataclass(frozen=True)
class TtsStreamResult:
    """Container for incremental synthesis results."""

    streamed: bool
    audio: bytes | None = None


class TtsStreamError(Exception):
    """Raised when a chunk-level TTS request fails."""

    def __init__(self, message: str, chunk_preview: str, voice: str) -> None:
        super().__init__(message)
        self.chunk_preview = chunk_preview
        self.voice = voice


class EmptyTtsAudioError(TtsStreamError):
    """Raised when Voxtral accepts a request but yields no audio."""


class VoxtralEventHandler(AsyncEventHandler):
    """Bridge Wyoming TTS events to the Voxtral streaming API."""

    def __init__(
        self,
        *args,
        info: Info,
        tts_client: TtsStreamClient,
        sample_rate: int,
        tts_streaming_min_words: int | None = None,
        tts_streaming_max_chars: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._wyoming_info = info
        self._tts_client = tts_client
        self._sample_rate = sample_rate
        self._tts_streaming_min_words = tts_streaming_min_words
        self._tts_streaming_max_chars = tts_streaming_max_chars

        self._last_event_type: str | None = None
        self._event_counter = 0

        self._synthesis_buffer: list[str] = []
        self._synthesis_voice: SynthesizeVoice | None = None
        self._is_synthesizing = False

        self._text_accumulator = ""
        self._pysbd_segmenters: dict[str, pysbd.Segmenter] = {}
        self._audio_started = False
        self._current_timestamp = 0.0

        self._tts_semaphore = asyncio.Semaphore(TTS_CONCURRENT_REQUESTS)
        self._allow_streaming_task_id: str | None = None
        self._active_tasks: set[asyncio.Task[object]] = set()
        self._stopping = False

    async def handle_event(self, event: Event) -> bool:
        """Handle incoming Wyoming events."""
        _LOGGER.debug("Incoming event type %s", event.type)

        if Synthesize.is_type(event.type):
            return await self._handle_synthesize(Synthesize.from_event(event))

        if SynthesizeStart.is_type(event.type):
            return await self._handle_synthesize_start(SynthesizeStart.from_event(event))

        if SynthesizeChunk.is_type(event.type):
            return await self._handle_synthesize_chunk(SynthesizeChunk.from_event(event))

        if SynthesizeStop.is_type(event.type):
            return await self._handle_synthesize_stop()

        if Describe.is_type(event.type):
            await self.write_event(self._wyoming_info.event())
            return True

        _LOGGER.info("Ignoring unhandled event type: %s", event.type)
        return True

    async def stop(self) -> None:
        """Cancel in-flight synthesis work when the handler is stopped."""
        self._stopping = True
        await self._cancel_active_tasks()
        await super().stop()

    def _get_voice(self, name: str | None = None) -> VoxtralVoice | None:
        """Get a configured TTS voice by name."""
        for program in self._wyoming_info.tts:
            for voice in program.voices:
                if not name or voice.name == name:
                    assert isinstance(
                        voice, VoxtralVoice
                    ), f"Voice {voice.name!r} is not a VoxtralVoice instance"
                    return voice
        return None

    def _is_tts_language_supported(self, language: str, voice: TtsVoice) -> bool:
        """Check whether a voice can serve the requested language."""
        return not voice.languages or language in voice.languages

    def _validate_tts_language(self, language: str | None, voice: TtsVoice) -> bool:
        """Validate a requested TTS language."""
        if language and not self._is_tts_language_supported(language, voice):
            _LOGGER.error(
                "Language %s is not supported for voice %s. Available languages: %s",
                language,
                voice.name,
                voice.languages,
            )
            return False
        return True

    def _validate_tts_voice_and_language(
        self,
        requested_voice: str | None,
        requested_language: str | None,
    ) -> VoxtralVoice | None:
        """Resolve and validate a Wyoming voice selection."""
        voice = self._get_voice(requested_voice)
        if not voice:
            self._log_unsupported_voice(requested_voice)
            return None

        if not self._validate_tts_language(requested_language, voice):
            return None

        return voice

    def _log_unsupported_voice(self, requested_voice: str | None) -> None:
        """Log an unsupported voice request."""
        if requested_voice:
            available = [
                voice.name for program in self._wyoming_info.tts for voice in program.voices
            ]
            _LOGGER.error(
                "Voice %s is not supported. Available voices: %s", requested_voice, available
            )
        else:
            _LOGGER.error("No TTS voices are configured")

    def _get_pysbd_language(self, language: str | None) -> str:
        """Map a Wyoming language into a pySBD-compatible code."""
        if not language:
            return "en"

        base_language = language[:2].lower()
        try:
            pysbd.Segmenter(language=base_language)
        except (KeyError, ValueError):
            _LOGGER.warning(
                "Language %s is not supported by pySBD, falling back to English", base_language
            )
            return "en"

        return base_language

    def _meets_min_criteria(self, text: str, min_words: int) -> bool:
        """Check whether a text chunk is large enough for synthesis."""
        return len(text.split()) >= min_words

    def _chunk_text_for_streaming(
        self,
        text: str,
        min_words: int | None = None,
        max_chars: int | None = None,
        language: str | None = None,
    ) -> list[str]:
        """Split text into sentence-aware streaming chunks."""
        if not text.strip():
            return []

        segmenter = pysbd.Segmenter(language=self._get_pysbd_language(language), clean=True)
        sentences = segmenter.segment(text)

        chunks: list[str] = []
        current_chunk = ""

        for sentence in sentences:
            candidate = f"{current_chunk} {sentence}".strip() if current_chunk else sentence

            if not max_chars and not min_words:
                # No constraints: emit each sentence individually.
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            elif max_chars and len(candidate) > max_chars and current_chunk:
                # Adding this sentence would exceed the character cap: flush current chunk
                # and start a new one. min_words is relaxed here — when max_chars forces
                # the split there is nowhere else for the accumulated text to go, so
                # emitting a short chunk is preferable to silently discarding it.
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = candidate
                # When only min_words is set: emit as soon as the threshold is reached
                # so that large texts are split into multiple chunks rather than
                # accumulating into a single request.
                if (
                    not max_chars
                    and min_words
                    and self._meets_min_criteria(current_chunk, min_words)
                ):
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks or [text]

    async def _handle_synthesize(self, synthesize: Synthesize) -> bool:
        """Handle one-shot synthesis requests."""
        if self._is_synthesizing:
            _LOGGER.debug("Ignoring standalone synthesize while streaming synthesis is active")
            return True

        if synthesize.voice:
            requested_voice = synthesize.voice.name
            requested_language = synthesize.voice.language
        else:
            requested_voice = None
            requested_language = None

        text = synthesize.text or ""
        voice = self._validate_tts_voice_and_language(requested_voice, requested_language)
        if not voice:
            return False

        synthesis_task = self._create_managed_task(
            self._stream_tts_audio(voice=voice, text=text, send_audio_start=True),
            name="voxtral_synthesize",
        )
        try:
            final_timestamp = await synthesis_task
        except asyncio.CancelledError:
            if self._stopping:
                return False
            raise
        if final_timestamp is None:
            return False

        await self.write_event(AudioStop(timestamp=int(final_timestamp)).event())
        _LOGGER.info("Successfully synthesized: %s", _truncate_for_log(text))
        return True

    async def _handle_synthesize_start(self, synthesize_start: SynthesizeStart) -> bool:
        """Start a streaming synthesis session."""
        self._synthesis_buffer = []
        self._is_synthesizing = True
        self._text_accumulator = ""
        self._pysbd_segmenters.clear()
        self._audio_started = False
        self._current_timestamp = 0.0

        self._synthesis_voice = synthesize_start.voice
        if not synthesize_start.voice:
            return True

        voice = self._validate_tts_voice_and_language(
            synthesize_start.voice.name,
            synthesize_start.voice.language,
        )
        if not voice:
            self._is_synthesizing = False
            return False

        return True

    async def _handle_synthesize_chunk(self, synthesize_chunk: SynthesizeChunk) -> bool:
        """Handle streamed text input and synthesize full sentences early."""
        if not self._is_synthesizing:
            _LOGGER.warning("Received synthesize-chunk without active synthesis")
            return False

        chunk_text = synthesize_chunk.text or ""
        self._synthesis_buffer.append(chunk_text)
        self._text_accumulator += chunk_text

        requested_language = self._synthesis_voice.language if self._synthesis_voice else None
        pysbd_language = self._get_pysbd_language(requested_language)
        if pysbd_language not in self._pysbd_segmenters:
            self._pysbd_segmenters[pysbd_language] = pysbd.Segmenter(
                language=pysbd_language, clean=True
            )

        sentences = list(self._pysbd_segmenters[pysbd_language].segment(self._text_accumulator))
        if len(sentences) <= 1:
            return True

        ready_sentences = sentences[:-1]
        self._text_accumulator = sentences[-1]
        return await self._process_ready_sentences(ready_sentences)

    async def _handle_synthesize_stop(self) -> bool:
        """Finish a streaming synthesis session."""
        if not self._is_synthesizing:
            _LOGGER.warning("Received synthesize-stop without active synthesis")
            return False

        self._is_synthesizing = False

        if self._text_accumulator.strip() and not await self._process_ready_sentences(
            [self._text_accumulator]
        ):
            return False

        self._synthesis_buffer = []
        self._synthesis_voice = None
        self._text_accumulator = ""
        self._pysbd_segmenters.clear()

        if self._audio_started:
            await self.write_event(AudioStop(timestamp=int(self._current_timestamp)).event())
            await self.write_event(SynthesizeStopped().event())
            self._audio_started = False
            self._current_timestamp = 0.0
            return True

        await self.write_event(SynthesizeStopped().event())
        return True

    async def _process_ready_sentences(self, sentences: list[str]) -> bool:
        """Start synthesis work for complete sentences and preserve output order."""
        if not sentences:
            return True
        if not self._synthesis_voice:
            _LOGGER.error("Cannot synthesize sentences without an active voice")
            return False

        voice = self._validate_tts_voice_and_language(
            self._synthesis_voice.name, self._synthesis_voice.language
        )
        if not voice:
            return await self._abort_synthesis()

        valid_sentences = [sentence for sentence in sentences if sentence.strip()]
        if not valid_sentences:
            return True

        if self._tts_streaming_min_words or self._tts_streaming_max_chars:
            valid_sentences = self._chunk_text_for_streaming(
                " ".join(valid_sentences),
                min_words=self._tts_streaming_min_words,
                max_chars=self._tts_streaming_max_chars,
                language=self._synthesis_voice.language,
            )

        synthesis_tasks = [
            (
                f"sentence_{index}",
                self._create_managed_task(
                    self._get_tts_audio_stream(sentence, voice, task_id=f"sentence_{index}"),
                    name=f"voxtral_sentence_{index}",
                ),
            )
            for index, sentence in enumerate(valid_sentences)
        ]

        for index, (task_id, task) in enumerate(synthesis_tasks):
            self._allow_streaming_task_id = task_id
            sentence_preview = _truncate_for_log(valid_sentences[index], 50)

            try:
                result = await task
            except EmptyTtsAudioError as err:
                _LOGGER.warning(
                    "Skipping sentence %d (%s) with voice %s after empty audio response",
                    index + 1,
                    err.chunk_preview,
                    err.voice,
                )
                continue
            except TtsStreamError as err:
                _LOGGER.error(
                    "Failed to synthesize sentence %d (%s) with voice %s: %s",
                    index + 1,
                    err.chunk_preview,
                    err.voice,
                    err,
                )
                return await self._abort_synthesis()
            except asyncio.CancelledError:
                if self._stopping:
                    return False
                raise
            except Exception as err:  # pragma: no cover - defensive boundary
                _LOGGER.exception(
                    "Unexpected error while synthesizing %s: %s", sentence_preview, err
                )
                return await self._abort_synthesis()
            finally:
                self._allow_streaming_task_id = None

            if result.streamed:
                continue

            if not result.audio:
                _LOGGER.error("Buffered synthesis returned no audio for %s", sentence_preview)
                return await self._abort_synthesis()

            timestamp = await self._stream_audio_to_wyoming(
                result.audio,
                is_first_chunk=(not self._audio_started),
                start_timestamp=self._current_timestamp,
            )
            if timestamp is None:
                return await self._abort_synthesis()

            self._current_timestamp = timestamp
            self._audio_started = True

        return True

    async def _abort_synthesis(self) -> bool:
        """Abort the active synthesis session and emit stop events."""
        await self._cancel_active_tasks()

        if self._audio_started:
            await self.write_event(AudioStop(timestamp=int(self._current_timestamp)).event())

        await self.write_event(SynthesizeStopped().event())

        self._audio_started = False
        self._allow_streaming_task_id = None
        self._current_timestamp = 0.0
        self._is_synthesizing = False
        self._pysbd_segmenters.clear()
        self._synthesis_buffer = []
        self._synthesis_voice = None
        self._text_accumulator = ""
        return False

    def _create_managed_task(
        self,
        coroutine: Coroutine[Any, Any, T],
        *,
        name: str,
    ) -> asyncio.Task[T]:
        """Create and track an asyncio task tied to this handler."""
        task = asyncio.create_task(coroutine, name=name)
        self._active_tasks.add(cast(asyncio.Task[object], task))
        task.add_done_callback(self._active_tasks.discard)
        return task

    async def _cancel_active_tasks(self) -> None:
        """Cancel all tracked tasks except the current one."""
        current_task = asyncio.current_task()
        tasks_to_cancel = [
            task for task in self._active_tasks if task is not current_task and not task.done()
        ]
        if not tasks_to_cancel:
            return

        for task in tasks_to_cancel:
            task.cancel()

        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                timeout=5.0,
            )
        except TimeoutError:
            _LOGGER.warning("Timed out waiting for %d task(s) to cancel", len(tasks_to_cancel))

    async def _get_tts_audio_stream(
        self,
        text: str,
        voice: VoxtralVoice,
        task_id: str | None = None,
    ) -> TtsStreamResult:
        """Fetch audio for a single synthesis chunk, streaming or buffering as needed."""
        chunk_preview = _truncate_for_log(text, 50)

        try:
            should_stream = task_id is not None and task_id == self._allow_streaming_task_id

            if should_stream:
                timestamp = await self._stream_tts_audio_incremental(text, voice)
                if timestamp is None:
                    raise TtsStreamError(
                        "Voxtral returned no audio while streaming", chunk_preview, voice.name
                    )
                return TtsStreamResult(streamed=True)

            chunks: list[bytes] = []
            for attempt in range(TTS_EMPTY_AUDIO_RETRY_ATTEMPTS + 1):
                chunks.clear()
                async with self._tts_semaphore:
                    async for chunk in self._tts_client.stream_speech(
                        model=voice.model_name,
                        text=text,
                        voice_id=voice.voice_id,
                        reference_audio_b64=voice.reference_audio_b64,
                    ):
                        if chunk:
                            chunks.append(chunk)

                audio_data = b"".join(chunks)
                if audio_data:
                    return TtsStreamResult(streamed=False, audio=audio_data)

                if attempt < TTS_EMPTY_AUDIO_RETRY_ATTEMPTS:
                    _LOGGER.warning(
                        "Retrying empty audio response for %s with voice %s (%d/%d)",
                        chunk_preview,
                        voice.name,
                        attempt + 1,
                        TTS_EMPTY_AUDIO_RETRY_ATTEMPTS,
                    )
                    continue

                raise EmptyTtsAudioError("Voxtral returned empty audio", chunk_preview, voice.name)

            raise AssertionError("Empty audio retry loop exited without returning or raising")

        except TtsStreamError:
            raise
        except Exception as exc:
            raise TtsStreamError(
                "Unexpected error while retrieving TTS audio", chunk_preview, voice.name
            ) from exc

    async def _stream_tts_audio_incremental(self, text: str, voice: VoxtralVoice) -> float | None:
        """Stream one chunk directly to Wyoming and update timestamp continuity."""
        timestamp = await self._stream_tts_audio(
            voice=voice,
            text=text,
            send_audio_start=(not self._audio_started),
            start_timestamp=self._current_timestamp,
        )
        if timestamp is not None:
            self._current_timestamp = timestamp
            self._audio_started = True
        return timestamp

    async def _stream_tts_audio(
        self,
        *,
        voice: VoxtralVoice,
        text: str,
        send_audio_start: bool,
        start_timestamp: float = 0.0,
    ) -> float | None:
        """Stream Voxtral PCM audio directly to Wyoming."""
        timestamp = start_timestamp
        received_audio = False

        try:
            async with self._tts_semaphore:
                async for audio_data in self._tts_client.stream_speech(
                    model=voice.model_name,
                    text=text,
                    voice_id=voice.voice_id,
                    reference_audio_b64=voice.reference_audio_b64,
                ):
                    if not audio_data:
                        continue

                    if send_audio_start:
                        await self.write_event(
                            AudioStart(
                                rate=self._sample_rate,
                                width=DEFAULT_AUDIO_WIDTH,
                                channels=DEFAULT_AUDIO_CHANNELS,
                            ).event()
                        )
                        send_audio_start = False

                    received_audio = True
                    await self.write_event(
                        AudioChunk(
                            audio=audio_data,
                            rate=self._sample_rate,
                            width=DEFAULT_AUDIO_WIDTH,
                            channels=DEFAULT_AUDIO_CHANNELS,
                            timestamp=int(timestamp),
                        ).event()
                    )

                    sample_count = len(audio_data) // DEFAULT_AUDIO_WIDTH
                    timestamp += (sample_count / self._sample_rate) * 1000

        except Exception as exc:
            _LOGGER.exception("Error streaming TTS audio: %s", exc)
            return None

        return timestamp if received_audio else None

    async def _stream_audio_to_wyoming(
        self,
        audio_data: bytes,
        *,
        is_first_chunk: bool,
        start_timestamp: float,
    ) -> float | None:
        """Write buffered PCM audio to Wyoming."""
        try:
            timestamp = start_timestamp

            if is_first_chunk:
                await self.write_event(
                    AudioStart(
                        rate=self._sample_rate,
                        width=DEFAULT_AUDIO_WIDTH,
                        channels=DEFAULT_AUDIO_CHANNELS,
                    ).event()
                )

            if audio_data:
                await self.write_event(
                    AudioChunk(
                        audio=audio_data,
                        rate=self._sample_rate,
                        width=DEFAULT_AUDIO_WIDTH,
                        channels=DEFAULT_AUDIO_CHANNELS,
                        timestamp=int(timestamp),
                    ).event()
                )
                sample_count = len(audio_data) // DEFAULT_AUDIO_WIDTH
                timestamp += (sample_count / self._sample_rate) * 1000

            return timestamp
        except Exception as exc:
            _LOGGER.exception("Error writing buffered audio to Wyoming: %s", exc)
            return None

    async def write_event(self, event: Event) -> None:
        """Add lightweight event logging around the base Wyoming writer."""
        type_changed = self._last_event_type != event.type
        if type_changed:
            self._last_event_type = event.type
            self._event_counter = 1
        else:
            self._event_counter += 1

        # Log all non-audio-chunk events; for audio-chunk, only log the first
        # occurrence per run to avoid flooding the log during streaming.
        if event.type != "audio-chunk" or self._event_counter == 1:
            _LOGGER.debug("Outgoing event type %s", event.type)

        await super().write_event(event)
