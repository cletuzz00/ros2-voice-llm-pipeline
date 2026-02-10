import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI
import whisper

# Reachy 2
from reachy2_sdk import ReachySDK

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

# LOAD ENV + OPENAI CLIENT

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Ensure .env is next to this script.")

openai_client = OpenAI(api_key=api_key)

# LOAD WHISPER
print("[INFO] Loading Whisper...")
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
print(f"[INFO] Whisper loaded: {WHISPER_MODEL_NAME}")

# AUDIO SETUP
sd.default.device = (INPUT_DEVICE_INDEX, OUTPUT_DEVICE_INDEX)
sd.default.samplerate = SAMPLE_RATE

# INITIALIZE REACHY2 CONNECTION
print(f"[INFO] Connecting to Reachy2 at {REACHY2_IP}")
reachy = None
try:
    reachy = ReachySDK(host=REACHY2_IP)
    if not reachy.is_connected():
        raise ConnectionError("Could not connect to Reachy 2.")
    print("[INFO] Reachy2 connected successfully")
except ImportError:
    print("[ERROR] reachy2-sdk not installed. Install with: pip install reachy2-sdk")
    exit(1)
except Exception as e:
    print(f"[ERROR] Failed to connect to Reachy2: {e}")
    exit(1)

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

# Keep conversation history (last K turns)
conversation_memory = []  # list of {"role": "...", "content": "..."}
MAX_TURNS_TO_KEEP = 10

def add_to_memory(role: str, content: str):
    conversation_memory.append({"role": role, "content": content})
    if len(conversation_memory) > 2 * MAX_TURNS_TO_KEEP + 1:
        del conversation_memory[:2]

# TIMER-BASED LISTENING
def record_audio(seconds: float = RECORD_SECONDS) -> np.ndarray:
    print(f"\nListening for {seconds:.1f}s...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocking=True
    )
    audio = np.squeeze(audio).astype(np.float32)
    print(" Done.")
    return audio

# TRANSCRIBE (ENGLISH ONLY)
def transcribe(audio: np.ndarray) -> str:
    print(" Transcribing...")
    result = whisper_model.transcribe(
        audio,
        language="en",
        task="transcribe",
        fp16=False
    )
    text = (result.get("text") or "").strip()
    print("You said:", text)
    return text

# OPENAI CHAT COMPLETIONS (e.g. ChatGPT 5.2 / gpt-4o)
def llm_response(user_msg: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT.strip()}]
    for item in conversation_memory[-2 * MAX_TURNS_TO_KEEP:]:
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": user_msg})

    response = openai_client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
    )
    reply = (response.choices[0].message.content or "").strip()
    return reply

# OPENAI TTS SPEAK ON REACHY2
def speak_on_reachy2(text: str) -> float:
    """Generate TTS audio and play on Reachy2 robot speaker. Returns TTS duration."""
    if not text:
        return 0.0

    print(f"Speaking on Reachy2: {text}")

    response = openai_client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        instructions="Speak clearly and naturally.",
        response_format="wav"
    )

    audio_file = "reachy_speech.wav"
    tts_duration = 0.0

    try:
        with open(audio_file, "wb") as f:
            f.write(response.read())

        audio_data, sample_rate = sf.read(audio_file)
        tts_duration = len(audio_data) / sample_rate

        print(f"TTS duration: {tts_duration:.2f}s")
        print(f"Uploading to robot: {audio_file}")
        reachy.audio.upload_audio_file(audio_file)

        print("Playing on robot speaker...")
        reachy.audio.play_audio_file(audio_file)

        time.sleep(tts_duration)
        print("Audio playback complete")

    except Exception as e:
        print(f"[ERROR] Failed to play audio on Reachy2: {e}")
        raise
    finally:
        try:
            if os.path.exists(audio_file):
                os.remove(audio_file)
        except Exception:
            pass

    return tts_duration

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
            if lower in ["exit", "quit", "stop"]:
                goodbye = "Shutting down. Goodbye."
                speak_on_reachy2(goodbye)
                break

            add_to_memory("user", text)

            try:
                reply = llm_response(text)
            except Exception as e:
                reply = "I had trouble reaching the language model. Please check that your local LLM is running."
                print("[ERROR] LLM call failed:", e)

            if not reply:
                reply = "I'm not sure I caught that. Could you repeat it?"

            add_to_memory("assistant", reply)
            print("Assistant:", reply)
            speak_on_reachy2(reply)

            # Delay to prevent mic picking up Reachy's own audio
            time.sleep(1.0)

        elapsed = time.time() - cycle_start
        remaining = CYCLE_SECONDS - elapsed
        if remaining > 0:
            time.sleep(remaining)

if __name__ == "__main__":
    main()
