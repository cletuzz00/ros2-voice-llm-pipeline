import os
import time
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI

from voice_pipeline.audio_input import record_fixed
from voice_pipeline.llm_client import ChatConfig, ChatSession
from voice_pipeline.stt_whisper import WhisperSTT
from voice_pipeline.reachy_connector import ReachyConfig, ReachyAudioConnector
from voice_pipeline.tts_output import speak_local

# REACHY2 ROBOT CONFIG
REACHY2_IP = "192.168.50.241"

# auto pick device indices with sounddevice
# Setting to None uses system default devices
INPUT_DEVICE_INDEX = None   # System default microphone
OUTPUT_DEVICE_INDEX = None  # System default speakers

# Print what devices will be used
if INPUT_DEVICE_INDEX is None and OUTPUT_DEVICE_INDEX is None:
    default_in, default_out = sd.default.device
    print(f"[INFO] Using system default audio devices:")
    print(f"  Input: {sd.query_devices(default_in)['name']}")
    print(f"  Output: {sd.query_devices(default_out)['name']}")
elif INPUT_DEVICE_INDEX is not None and OUTPUT_DEVICE_INDEX is not None:
    print(f"[INFO] Using specified audio devices:")
    print(f"  Input: {sd.query_devices(INPUT_DEVICE_INDEX)['name']}")
    print(f"  Output: {sd.query_devices(OUTPUT_DEVICE_INDEX)['name']}")
else:
    if INPUT_DEVICE_INDEX is not None:
        print(f"[INFO] Using specified input device:")
        print(f"  Input: {sd.query_devices(INPUT_DEVICE_INDEX)['name']}")
    if OUTPUT_DEVICE_INDEX is not None:
        print(f"[INFO] Using specified output device:")
        print(f"  Output: {sd.query_devices(OUTPUT_DEVICE_INDEX)['name']}")

# AUDIO SETTINGS
SAMPLE_RATE = 16000          # Whisper-friendly
CHANNELS = 1                 # mic input channel
RECORD_SECONDS = 5.0         # length of each capture window
CYCLE_SECONDS = 4.0          # how often to run a new cycle (>= RECORD_SECONDS)

WHISPER_MODEL_NAME = "base"  # Whisper base model
# OpenAI Chat model (e.g. gpt-5.2 when available; use gpt-4o as fallback)
OPENAI_CHAT_MODEL = "gpt-4o"

TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"
TTS_SR_HINT = 24000          # typical output SR for OpenAI TTS wav

# OUTPUT BACKEND TOGGLE
# True  -> use Reachy connector
# False -> use local speakers via OpenAI TTS
USE_REACHY = False

# Exit keywords that will stop the main loop when heard in the transcript
EXIT_KEYWORDS = ("exit", "quit", "stop", "bye", "goodbye", "bye bye")

# LOAD ENV + OPENAI CLIENT

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Ensure .env is next to this script.")

openai_client = OpenAI(api_key=api_key)

# LOAD WHISPER
print("[INFO] Loading Whisper...")
stt = WhisperSTT(WHISPER_MODEL_NAME)
print(f"[INFO] Whisper loaded: {WHISPER_MODEL_NAME}")

 # AUDIO SETUP
sd.default.device = (INPUT_DEVICE_INDEX, OUTPUT_DEVICE_INDEX)
sd.default.samplerate = SAMPLE_RATE

# INITIALIZE REACHY2 CONNECTION (via connector) – only when using Reachy backend
if USE_REACHY:
    print(f"[INFO] Connecting to Reachy2 at {REACHY2_IP}")
    try:
        reachy_config = ReachyConfig(host=REACHY2_IP)
        reachy_connector = ReachyAudioConnector(reachy_config)
        print("[INFO] Reachy2 connected successfully")
    except ImportError:
        print("[ERROR] reachy2-sdk not installed. Install with: pip install reachy2-sdk")
        exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to connect to Reachy2: {e}")
        exit(1)
else:
    reachy_connector = None  # type: ignore[assignment]

# SYSTEM PROMPT (general robot assistant; replace with reachy2 trivia prompt if you want quiz behavior)
SYSTEM_PROMPT = """
You are the conversational brain for a robot voice pipeline used in a Human–Robot Interaction project.

Context:
- The user speaks into a microphone.
- Whisper transcribes the speech to text.
- You (the LLM) generate a helpful, spoken response.
- A text-to-speech system will read your response aloud to the user via the robot's speaker.

Behavior rules:
- Keep responses concise (1–4 sentences unless the user asks for detail).
- Speak naturally, like a friendly robot assistant.
- If the transcription seems incomplete or unclear, ask a short follow-up question.
- Do not mention internal tool names (Whisper, Ollama, OpenAI TTS, ROS).
- If the user says "stop", "quit", or "exit", respond with a short goodbye.

Topic focus:
- You are a Data Science Master's student at the University of Michigan Tech.
"""

# LLM client (conversation memory handled inside ChatSession)
MAX_TURNS_TO_KEEP = 10
chat_config = ChatConfig(
    model=OPENAI_CHAT_MODEL,
    system_prompt=SYSTEM_PROMPT,
    max_turns=MAX_TURNS_TO_KEEP,
)
chat_session = ChatSession(openai_client, chat_config)

def record_audio(seconds: float = RECORD_SECONDS) -> np.ndarray:
    """
    Timer-based listening using the shared audio_input helper.
    """
    return record_fixed(seconds, SAMPLE_RATE, CHANNELS)


def transcribe(audio: np.ndarray) -> str:
    """
    Run Whisper STT on the given audio buffer.
    """
    print(" Transcribing...")
    text = stt.transcribe(audio, language="en")
    print("You said:", text)
    return text


def llm_response(user_msg: str) -> str:
    """
    Delegate to the shared ChatSession for reply generation.
    """
    return chat_session.generate_reply(user_msg)


def speak_on_reachy2(text: str) -> float:
    """
    Synthesize TTS via OpenAI and play it on Reachy2 using the connector.
    Returns the approximate TTS duration.
    """
    return reachy_connector.speak_via_openai_tts(
        text=text,
        client=openai_client,
        model=TTS_MODEL,
        voice=TTS_VOICE,
    )


# Bind speak() once based on USE_REACHY
if USE_REACHY:
    speak = speak_on_reachy2
else:
    speak = lambda text: speak_local(text, openai_client, TTS_MODEL, TTS_VOICE)

# MAIN LOOP
def main():
    print("\nREADY — Timer-based voice pipeline with Reachy2 audio output (GPT LLM).")
    print(f"Cycle every {CYCLE_SECONDS}s, recording {RECORD_SECONDS}s each cycle.\n")

    while True:
        cycle_start = time.time()

        audio = record_audio(RECORD_SECONDS)
        text = transcribe(audio)

        if not text:
            print("[INFO] Empty transcription. Skipping.")
        else:
            lower = text.lower().strip()
            if any(kw in lower for kw in EXIT_KEYWORDS):
                goodbye = "Shutting down. Goodbye."
                speak(goodbye)
                break

            try:
                reply = llm_response(text)
            except Exception as e:
                reply = "I had trouble reaching the language model. Please check that your local LLM is running."
                print("[ERROR] LLM call failed:", e)

            if not reply:
                reply = "I'm not sure I caught that. Could you repeat it?"

            print("Assistant:", reply)
            speak(reply)

            # Delay to prevent mic picking up Reachy's own audio
            time.sleep(1.0)

        elapsed = time.time() - cycle_start
        remaining = CYCLE_SECONDS - elapsed
        if remaining > 0:
            time.sleep(remaining)

if __name__ == "__main__":
    main()
