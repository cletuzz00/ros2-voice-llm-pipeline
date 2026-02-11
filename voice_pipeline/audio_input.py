import numpy as np
import sounddevice as sd


def record_fixed(seconds: float, sample_rate: int, channels: int = 1) -> np.ndarray:
    """
    Record a fixed-duration mono audio clip from the default (or configured)
    microphone and return it as a float32 numpy array at the given sample rate.

    This is essentially the common pattern used in the existing scripts, but
    extracted into a reusable helper.
    """
    print(f"\nRecording for {seconds:.1f}s...")
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        blocking=True,
    )
    audio = np.squeeze(audio).astype(np.float32)
    print(" Done.")
    return audio


