from __future__ import annotations

from pathlib import Path
from typing import Union, Optional

import sounddevice as sd
import soundfile as sf
from openai import OpenAI


PathLike = Union[str, Path]


def synthesize_to_file(
    text: str,
    client: OpenAI,
    model: str,
    voice: str,
    out_path: PathLike,
    instructions: Optional[str] = None,
    response_format: str = "wav",
) -> Path:
    """
    Generate speech audio for the given text and write it to a file.

    Returns the Path to the written file.
    """
    out_path = Path(out_path)

    if not text:
        # Ensure the caller always gets a path back
        if not out_path.exists():
            out_path.write_bytes(b"")
        return out_path

    kwargs = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": response_format,
    }
    if instructions:
        kwargs["instructions"] = instructions

    resp = client.audio.speech.create(**kwargs)
    with out_path.open("wb") as f:
        f.write(resp.read())

    return out_path


def speak_local(
    text: str,
    client: OpenAI,
    model: str,
    voice: str,
    out_wav: PathLike = "tts_output.wav",
) -> None:
    """
    Convenience wrapper: synthesize speech then play it on the local speakers.
    """
    if not text:
        return

    wav_path = synthesize_to_file(text, client, model, voice, out_wav)
    audio, sr = sf.read(wav_path)
    sd.play(audio, sr)
    sd.wait()


