import os
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class OpenAITTSNode(Node):
    """
    Subscribe: /robot_reply (String)
    Speak via OpenAI TTS (gpt-4o-mini-tts)
    """

    def __init__(self):
        super().__init__('openai_tts_node')

        self.declare_parameter('env_path', os.path.expanduser('~/.env'))
        self.declare_parameter('voice', 'alloy')
        self.declare_parameter('model', 'gpt-4o-mini-tts')
        self.declare_parameter('output_device', -1)  # -1 = default output

        env_path = str(self.get_parameter('env_path').value)
        load_dotenv(env_path)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found. Set it in ~/.env or export it.")

        self.client = OpenAI(api_key=api_key)

        self.voice = str(self.get_parameter('voice').value)
        self.model = str(self.get_parameter('model').value)
        self.output_device = int(self.get_parameter('output_device').value)

        self.sub = self.create_subscription(String, '/robot_reply', self.on_reply, 10)
        self.get_logger().info(f"OpenAI TTS ready: model={self.model} voice={self.voice}")

    def speak(self, text: str):
        if not text:
            return

        # Generate audio bytes
        resp = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text
        )

        # Write to temp wav and play
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
            f.write(resp.read())

        audio, sr = sf.read(wav_path)
        if self.output_device >= 0:
            sd.default.device = (None, self.output_device)

        sd.play(audio, sr)
        sd.wait()

        try:
            os.remove(wav_path)
        except Exception:
            pass

    def on_reply(self, msg: String):
        text = (msg.data or "").strip()
        if not text:
            return
        self.get_logger().info(f"🔊 Speaking reply ({len(text)} chars)")
        try:
            self.speak(text)
        except Exception as e:
            self.get_logger().error(f"TTS failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = OpenAITTSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

