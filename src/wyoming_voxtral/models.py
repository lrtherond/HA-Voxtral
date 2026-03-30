"""Shared data models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from wyoming.info import TtsVoice


class VoiceKind(StrEnum):
    """The backing source for a Wyoming voice."""

    SAVED = "saved"
    REFERENCE = "reference"


@dataclass(frozen=True, slots=True)
class SavedVoice:
    """Saved Mistral voice metadata."""

    voice_id: str
    display_name: str
    description: str
    languages: tuple[str, ...]
    raw_name: str
    slug: str | None = None


@dataclass(frozen=True, slots=True)
class ReferenceVoice:
    """Reference-audio voice loaded from disk."""

    display_name: str
    description: str
    languages: tuple[str, ...]
    source_path: Path
    reference_audio_b64: str


class VoxtralVoice(TtsVoice):
    """Wyoming TTS voice with enough metadata to call Voxtral."""

    def __init__(
        self,
        *,
        model_name: str,
        voice_kind: VoiceKind,
        voice_id: str | None = None,
        reference_audio_b64: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if (voice_id is None) == (reference_audio_b64 is None):
            raise ValueError("Exactly one of voice_id or reference_audio_b64 must be provided")

        self.model_name = model_name
        self.voice_kind = voice_kind
        self.voice_id = voice_id
        self.reference_audio_b64 = reference_audio_b64
