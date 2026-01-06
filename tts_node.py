import time
import numpy as np
import sounddevice as sd
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

        self.device_index = int(self.get_parameter('device_index').value)
        self.sample_rate = int(self.get_parameter('sample_rate').value)
        self.record_seconds = float(self.get_parameter('record_seconds').value)
        self.cycle_seconds = float(self.get_parameter('cycle_seconds').value)
        self.model_name = str(self.get_parameter('model_name').value)

        # Publisher
        self.pub = self.create_publisher(String, '/speech_text', 10)

        # Load Whisper once
        self.get_logger().info(f"Loading Whisper model: {self.model_name}")
        self.model = whisper.load_model(self.model_name)
        self.get_logger().info("Whisper loaded.")

        # Timer
        self.timer = self.create_timer(self.cycle_seconds, self.tick)
        self.get_logger().info(
            f"Ready. Recording {self.record_seconds}s every {self.cycle_seconds}s "
            f"at {self.sample_rate} Hz (device_index={self.device_index})."
        )

    def record_audio(self) -> np.ndarray:
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

