import time
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class WhisperSTTNode(Node):
    """
    Timer-based mic recording -> Whisper -> publish transcript to /speech_text
    """

    def __init__(self):
        super().__init__('whisper_stt_node')

        # ROS params
        self.declare_parameter('device_index', -1)          # -1 = default input
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('record_seconds', 4.0)
        self.declare_parameter('cycle_seconds', 5.0)
        self.declare_parameter('model_name', 'base')
        self.declare_parameter('use_audio_file', False)      # Use audio file instead of mic
        self.declare_parameter('audio_file_path', '')        # Path to audio file

        self.device_index = int(self.get_parameter('device_index').value)
        self.sample_rate = int(self.get_parameter('sample_rate').value)
        self.record_seconds = float(self.get_parameter('record_seconds').value)
        self.cycle_seconds = float(self.get_parameter('cycle_seconds').value)
        self.model_name = str(self.get_parameter('model_name').value)
        self.use_audio_file = bool(self.get_parameter('use_audio_file').value)
        self.audio_file_path = str(self.get_parameter('audio_file_path').value)

        # Publisher
        self.pub = self.create_publisher(String, '/speech_text', 10)

        # Load Whisper once
        self.get_logger().info(f"Loading Whisper model: {self.model_name}")
        self.model = whisper.load_model(self.model_name)
        self.get_logger().info("Whisper loaded.")

        # Timer
        self.timer = self.create_timer(self.cycle_seconds, self.tick)
        if self.use_audio_file and self.audio_file_path:
            self.get_logger().info(
                f"Ready. Using audio file: {self.audio_file_path} "
                f"every {self.cycle_seconds}s (model={self.model_name})."
            )
        else:
            self.get_logger().info(
                f"Ready. Recording {self.record_seconds}s every {self.cycle_seconds}s "
                f"at {self.sample_rate} Hz (device_index={self.device_index})."
            )

    def record_audio(self) -> np.ndarray:
        if self.use_audio_file and self.audio_file_path:
            # Load audio from file
            if not os.path.exists(self.audio_file_path):
                self.get_logger().error(f"Audio file not found: {self.audio_file_path}")
                return np.array([], dtype=np.float32)
            
            self.get_logger().info(f"📁 Loading audio from file: {self.audio_file_path}")
            audio, sr = sf.read(self.audio_file_path)
            
            # Convert stereo to mono if needed
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sr != self.sample_rate:
                self.get_logger().info(f"Resampling from {sr} Hz to {self.sample_rate} Hz...")
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                except ImportError:
                    self.get_logger().error(
                        f"librosa not available. Cannot resample from {sr} Hz to {self.sample_rate} Hz. "
                        "Please install librosa: pip install librosa"
                    )
                    raise RuntimeError("librosa required for audio file resampling")
            
            audio = audio.astype(np.float32)
            self.get_logger().info("📁 Audio loaded.")
            return audio
        else:
            # Record from microphone
            frames = int(self.record_seconds * self.sample_rate)
            self.get_logger().info("🎤 Recording...")
            audio = sd.rec(
                frames,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=None if self.device_index < 0 else self.device_index,
                blocking=True
            )
            audio = np.squeeze(audio).astype(np.float32)
            self.get_logger().info("🎤 Done.")
            return audio

    def transcribe(self, audio: np.ndarray) -> str:
        self.get_logger().info("📝 Transcribing (English only)...")
        result = self.model.transcribe(
            audio,
            language="en",
            task="transcribe",
            fp16=False
        )
        return (result.get("text") or "").strip()

    def tick(self):
        try:
            audio = self.record_audio()
            text = self.transcribe(audio)

            if not text:
                self.get_logger().info("Empty transcription; skipping publish.")
                return

            msg = String()
            msg.data = text
            self.pub.publish(msg)
            self.get_logger().info(f'Published /speech_text: "{text}"')

        except Exception as e:
            self.get_logger().error(f"STT failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = WhisperSTTNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

