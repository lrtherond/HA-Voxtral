"""Voice catalog and Wyoming info helpers."""

from __future__ import annotations

from typing import cast

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice

from .const import ATTRIBUTION_NAME, ATTRIBUTION_URL, __version__
from .models import ReferenceVoice, SavedVoice, VoiceKind, VoxtralVoice


def create_tts_voices(
    *,
    model_name: str,
    saved_voices: list[SavedVoice],
    reference_voices: list[ReferenceVoice],
) -> list[VoxtralVoice]:
    """Convert discovered voices into Wyoming voice definitions."""
    attribution = Attribution(name=ATTRIBUTION_NAME, url=ATTRIBUTION_URL)
    voices: list[VoxtralVoice] = []

    for saved_voice in saved_voices:
        voices.append(
            VoxtralVoice(
                name=saved_voice.display_name,
                description=saved_voice.description,
                attribution=attribution,
                installed=True,
                languages=list(saved_voice.languages),
                model_name=model_name,
                reference_audio_b64=None,
                version=None,
                voice_id=saved_voice.voice_id,
                voice_kind=VoiceKind.SAVED,
            )
        )

    for reference_voice in reference_voices:
        voices.append(
            VoxtralVoice(
                name=reference_voice.display_name,
                description=reference_voice.description,
                attribution=attribution,
                installed=True,
                languages=list(reference_voice.languages),
                model_name=model_name,
                reference_audio_b64=reference_voice.reference_audio_b64,
                version=None,
                voice_id=None,
                voice_kind=VoiceKind.REFERENCE,
            )
        )

    return voices


def create_info(tts_voices: list[VoxtralVoice]) -> Info:
    """Build the Wyoming info response for a TTS-only server."""
    if not tts_voices:
        return Info(asr=[], tts=[])

    return Info(
        asr=[],
        tts=[
            TtsProgram(
                name="voxtral-tts",
                description="Mistral Voxtral TTS",
                attribution=Attribution(name=ATTRIBUTION_NAME, url=ATTRIBUTION_URL),
                installed=True,
                version=__version__,
                voices=cast(list[TtsVoice], list(tts_voices)),
                supports_synthesize_streaming=True,
            )
        ],
    )
