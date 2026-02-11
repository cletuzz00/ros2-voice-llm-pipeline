import os
import time
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import requests
import wave
from dotenv import load_dotenv
from openai import OpenAI
import whisper

# Reachy 2 Imports
from reachy2_sdk import ReachySDK

# REACHY2 ROBOT CONFIG
REACHY2_IP = "192.168.50.241"

# AUDIO SETTINGS
SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5.0
CYCLE_SECONDS = 4.0

WHISPER_MODEL_NAME = "base"
OLLAMA_MODEL = "mistral"
OLLAMA_URL = "http://localhost:11434/api/generate"

TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"

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

# INITIALIZE REACHY2 CONNECTION - CORRECT INITIALIZATION
print(f"[INFO] Connecting to Reachy2 at {REACHY2_IP}")
reachy = None
try:
    reachy = ReachySDK(host=REACHY2_IP)  # Only host parameter, no port!
    if not reachy.is_connected():
        raise ConnectionError("Could not connect to Reachy 2.")
    print("[INFO]  Reachy2 connected successfully")
except ImportError:
    print("[ERROR] reachy2-sdk not installed. Install with: pip install reachy2-sdk")
    exit(1)
except Exception as e:
    print(f"[ERROR] Failed to connect to Reachy2: {e}")
    exit(1)

# AUDIO SETUP
sd.default.samplerate = SAMPLE_RATE

# SYSTEM PROMPT
SYSTEM_PROMPT = """
You are the conversational brain for a robot voice pipeline used in a Human–Robot Interaction project.

Context:
- The user speaks into a microphone.
- Whisper transcribes the speech to text.
- You (the LLM) generate a helpful, spoken response.
- A text-to-speech system will read your response aloud to the user via the robot's speaker.

Behavior rules:
- Keep responses concise (1–4 sentences).
- Speak naturally, like a friendly robot assistant.
- If the transcription seems incomplete or unclear, ask a short follow-up question.
- If the user says "stop", "quit", or "exit", respond with a short goodbye.

Topic focus:
- You are playing a country trivia quiz game with the user.
- If no question is active, ask a new trivia question with multiple choice options (A, B, C).
- If a question was just asked, evaluate the user's answer (correct/incorrect) and briefly explain why.
- Keep track of the score in your responses (e.g., "That's correct! Score: 2/3").
- After evaluating an answer, ask the next question.
- Make it fun and encouraging!
"""

# Keep conversation history
conversation_memory = []
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

# TRANSCRIBE
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

# OLLAMA RESPONSE
def llm_response(user_msg: str, model: str = OLLAMA_MODEL) -> str:
    lines = []
    lines.append("SYSTEM:\n" + SYSTEM_PROMPT.strip() + "\n")

    for item in conversation_memory[-2 * MAX_TURNS_TO_KEEP:]:
        if item["role"] == "user":
            lines.append(f"User: {item['content']}")
        elif item["role"] == "assistant":
            lines.append(f"Assistant: {item['content']}")

    lines.append(f"User: {user_msg}")
    lines.append("Assistant:")

    prompt = "\n".join(lines)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    reply = r.json().get("response", "").strip()
    return reply

# OPENAI TTS SPEAK ON REACHY2 - Using wave file like your working code
def speak_on_reachy2(text: str) -> float:
    """Generate TTS audio and play on Reachy2 robot speaker. Returns TTS duration."""
    if not text:
        return 0.0

    print(f"🤖 Speaking on Reachy2: {text}")

    # Generate audio via OpenAI TTS
    response = openai_client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        instructions="Speak clearly and naturally.",
        response_format="wav"
    )

    # Write to temp wav file and play on Reachy2
    audio_file = "reachy_speech.wav"
    tts_duration = 0.0
    
    try:
        with open(audio_file, "wb") as f:
            f.write(response.read())

        # Calculate TTS duration BEFORE playing/deleting
        audio_data, sample_rate = sf.read(audio_file)
        tts_duration = len(audio_data) / sample_rate
        
        print(f"TTS duration: {tts_duration:.2f}s")

        print(f"Uploading to robot: {audio_file}")
        reachy.audio.upload_audio_file(audio_file)

        print(f"Playing on robot speaker...")
        reachy.audio.play_audio_file(audio_file)
        
        # Wait for playback based on actual TTS duration
        time.sleep(tts_duration)
        print("Audio playback complete")

    except Exception as e:
        print(f"[ERROR] Failed to play audio on Reachy2: {e}")
        raise
    finally:
        # Clean up temp file
        try:
            if os.path.exists(audio_file):
                os.remove(audio_file)
        except Exception:
            pass
    
    return tts_duration

# MAIN LOOP
def main():
    print("\nREADY — Timer-based voice pipeline with Reachy2 audio output.")
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
            
            # Add listener delay to prevent picking up Reachy's own audio
            time.sleep(1.0)

        # Timer pacing
        elapsed = time.time() - cycle_start
        remaining = CYCLE_SECONDS - elapsed
        if remaining > 0:
            time.sleep(remaining)

if __name__ == "__main__":
    main()