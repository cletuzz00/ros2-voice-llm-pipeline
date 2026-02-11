from __future__ import annotations

import numpy as np
import whisper


class WhisperSTT:
    """
    Thin wrapper around a Whisper model for English transcription.

    This class is intentionally unaware of microphones or file I/O – it just
    takes a mono float32 numpy array and returns a text transcription.
    """

    def __init__(self, model_name: str = "base") -> None:
        self.model_name = model_name
        self._model = whisper.load_model(model_name)

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        """
        Run Whisper transcription on a mono float32 waveform.
        """
        result = self._model.transcribe(
            audio,
            language=language,
            task="transcribe",
            fp16=False,
        )
        return (result.get("text") or "").strip()


