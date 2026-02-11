from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import soundfile as sf
from openai import OpenAI

from .tts_output import synthesize_to_file

try:
    from reachy2_sdk import ReachySDK
except ImportError as e:  # pragma: no cover - import-time guard
    ReachySDK = None  # type: ignore[assignment]
    _reachy_import_error = e
else:
    _reachy_import_error = None


PathLike = Union[str, Path]


@dataclass
class ReachyConfig:
    host: str = "192.168.50.241"


class ReachyAudioConnector:
    """
    Helper for playing audio on a Reachy 2 robot.

    It can either play an existing WAV file or synthesize speech via OpenAI TTS
    and then upload/play that file on the robot.
    """

    def __init__(self, config: ReachyConfig) -> None:
        if ReachySDK is None:
            raise ImportError(
                "reachy2-sdk not installed. Install with: pip install reachy2-sdk"
            ) from _reachy_import_error

        self._config = config
        self._reachy = ReachySDK(host=config.host)  # type: ignore[call-arg]
        if not self._reachy.is_connected():
            raise ConnectionError("Could not connect to Reachy 2.")

    def play_wav(self, path: PathLike) -> float:
        """
        Upload and play an existing WAV file on Reachy.

        Returns the approximate audio duration in seconds (based on the file).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        audio_data, sample_rate = sf.read(path)
        duration = len(audio_data) / sample_rate if sample_rate > 0 else 0.0

        self._reachy.audio.upload_audio_file(str(path))
        self._reachy.audio.play_audio_file(str(path))

        if duration > 0:
            time.sleep(duration)

        return duration

    def speak_via_openai_tts(
        self,
        text: str,
        client: OpenAI,
        model: str,
        voice: str,
        tmp_path: PathLike = "reachy_speech.wav",
    ) -> float:
        """
        Synthesize speech with OpenAI then play it on Reachy.

        Returns the approximate TTS duration in seconds.
        """
        if not text:
            return 0.0

        wav_path = synthesize_to_file(
            text=text,
            client=client,
            model=model,
            voice=voice,
            out_path=tmp_path,
            instructions="Speak clearly and naturally.",
            response_format="wav",
        )

        try:
            duration = self.play_wav(wav_path)
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

        return duration


