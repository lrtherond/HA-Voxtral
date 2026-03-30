from __future__ import annotations

import base64
import os
import struct

from wyoming_voxtral.utilities import load_dotenv, load_reference_voices, pcm_f32le_to_s16le


def test_load_reference_voices_reads_audio_files(tmp_path):
    voice_path = tmp_path / "Alice.wav"
    voice_path.write_bytes(b"voice-bytes")

    voices = load_reference_voices(tmp_path, ["en", "fr"])

    assert len(voices) == 1
    assert voices[0].display_name == "Alice"
    assert voices[0].languages == ("en", "fr")
    assert voices[0].reference_audio_b64 == base64.b64encode(b"voice-bytes").decode("ascii")


def test_load_dotenv_sets_missing_values(tmp_path, monkeypatch):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text('MISTRAL_API_KEY="test-key"\nVOXTRAL_MODEL=voxtral-mini-tts-2603\n')

    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("VOXTRAL_MODEL", raising=False)

    load_dotenv(dotenv_path)

    assert os.environ["MISTRAL_API_KEY"] == "test-key"
    assert os.environ["VOXTRAL_MODEL"] == "voxtral-mini-tts-2603"


def test_pcm_f32le_to_s16le_converts_expected_values():
    audio = struct.pack("<fff", -1.0, 0.0, 1.0)
    converted = pcm_f32le_to_s16le(audio)
    assert converted == struct.pack("<hhh", -32768, 0, 32767)
