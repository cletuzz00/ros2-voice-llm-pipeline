import requests

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


SYSTEM_PROMPT = """
You are the conversational brain for a robot voice pipeline used in a Human–Robot Interaction project.

Context:
- A microphone captures the user's speech.
- Speech is transcribed to text.
- You generate a helpful spoken response.
- A TTS engine will read your response aloud.

Behavior rules:
- Keep responses concise (1–4 sentences) unless the user asks for detail.
- If the input seems unclear, ask a short follow-up question.
- Do not mention internal tool names.
- If the user says stop/quit/exit, respond with a short goodbye.
""".strip()


class LLMNode(Node):
    """
    Subscribe: /speech_text (String)
    Publish:   /robot_reply (String)
    """

    def __init__(self):
        super().__init__('llm_node')

        self.declare_parameter('ollama_url', 'http://localhost:11434/api/generate')
        self.declare_parameter('model', 'mistral')
        self.declare_parameter('max_turns', 8)

        self.ollama_url = str(self.get_parameter('ollama_url').value)
        self.model = str(self.get_parameter('model').value)
        self.max_turns = int(self.get_parameter('max_turns').value)

        self.sub = self.create_subscription(String, '/speech_text', self.on_text, 10)
        self.pub = self.create_publisher(String, '/robot_reply', 10)

        self.memory = []  # list[str] like "User: ..." / "Assistant: ..."

        self.get_logger().info(f"LLM node ready. Using model={self.model} url={self.ollama_url}")

    def call_ollama(self, user_text: str) -> str:
        self.memory.append(f"User: {user_text}")
        # keep last N lines (roughly N turns)
        self.memory = self.memory[-2 * self.max_turns:]

        prompt = "SYSTEM:\n" + SYSTEM_PROMPT + "\n\n" + "\n".join(self.memory) + "\nAssistant:"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        r = requests.post(self.ollama_url, json=payload, timeout=120)
        r.raise_for_status()
        reply = (r.json().get("response") or "").strip()

        self.memory.append(f"Assistant: {reply}")
        self.memory = self.memory[-2 * self.max_turns:]
        return reply

    def on_text(self, msg: String):
        user_text = (msg.data or "").strip()
        if not user_text:
            return

        lower = user_text.lower().strip()
        if lower in ["stop", "quit", "exit"]:
            reply = "Shutting down. Goodbye."
        else:
            try:
                reply = self.call_ollama(user_text)
            except Exception as e:
                self.get_logger().error(f"Ollama call failed: {e}")
                reply = "I had trouble reaching the language model. Please check that it is running."

        out = String()
        out.data = reply
        self.pub.publish(out)
        self.get_logger().info(f'Published /robot_reply: "{reply}"')


def main(args=None):
    rclpy.init(args=args)
    node = LLMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()